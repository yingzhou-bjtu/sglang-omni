mod headers;
mod request_body;
mod response_body;

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::extract::{Extension, State};
use axum::http::header::{CONTENT_LENGTH, CONTENT_TYPE};
use axum::http::{HeaderValue, Method, Request, Response, Version};

use crate::config::Config;
use crate::error::{HttpFault, RouterError};
use crate::request_id::{CanonicalRequestId, REQUEST_ID_HEADER};
use crate::worker_pool::{AdmissionError, DispatchError, RequestLease, TrustDomain, WorkerPool};

use headers::{canonical_content_type, sanitize_response, validate_request};
use request_body::{DirectRequestBody, SharedUploadState, UploadState};
use response_body::DirectResponseBody;

pub(crate) const CHAT_PATH: &str = "/v1/chat/completions";

pub(crate) struct HttpGeneration {
    pool: Arc<WorkerPool>,
    client: reqwest::Client,
    trust: TrustDomain,
    streamed_max: u64,
    request_timeout: std::time::Duration,
}

impl HttpGeneration {
    pub(crate) fn build(config: &Config, pool: Arc<WorkerPool>) -> Result<Arc<Self>, RouterError> {
        let http_generation = &config.http_generation;
        let client = pool.generation_client();
        Ok(Arc::new(Self {
            client,
            pool,
            trust: TrustDomain::new(http_generation.trust_domain.clone()),
            streamed_max: http_generation.streamed_request_max_bytes,
            request_timeout: http_generation.request_timeout(),
        }))
    }

    pub(crate) fn is_ready(&self) -> bool {
        self.pool.generation_http_ready(&self.trust)
    }
}

pub(crate) async fn chat(
    State(generation): State<Arc<HttpGeneration>>,
    Extension(request_id): Extension<CanonicalRequestId>,
    request: Request<Body>,
) -> Response<Body> {
    match handle(generation, request, request_id.into_header_value()).await {
        Ok(response) => response,
        Err(fault) => fault.into_response(),
    }
}

async fn handle(
    generation: Arc<HttpGeneration>,
    request: Request<Body>,
    request_id: HeaderValue,
) -> Result<Response<Body>, HttpFault> {
    if request.method() != Method::POST {
        return Err(HttpFault::MethodNotAllowed);
    }
    if request.version() != Version::HTTP_11 {
        return Err(HttpFault::HttpVersionNotSupported);
    }
    if request.uri().path() != CHAT_PATH || request.uri().query().is_some() {
        return Err(HttpFault::MalformedRequest);
    }
    let deadline = tokio::time::Instant::now() + generation.request_timeout;
    let framing = validate_request(request.headers())?;
    let proof = generation
        .pool
        .content_blind_generation_http(&generation.trust)
        .ok_or(HttpFault::NoCompatibleWorker)?;
    if framing.content_length > generation.streamed_max {
        return Err(HttpFault::RequestBodyTooLarge);
    }
    let admission = generation.pool.try_admit().map_err(map_admission)?;
    let lease = proof.dispatch(admission).map_err(map_dispatch)?;
    relay_direct(
        generation,
        request.into_body(),
        framing.content_length,
        lease,
        request_id,
        deadline,
    )
    .await
}

async fn relay_direct(
    generation: Arc<HttpGeneration>,
    body: Body,
    length: u64,
    lease: RequestLease,
    request_id: HeaderValue,
    deadline: tokio::time::Instant,
) -> Result<Response<Body>, HttpFault> {
    let state: SharedUploadState = Arc::new(Mutex::new(UploadState::Incomplete));
    let direct = DirectRequestBody::new(
        body,
        length,
        generation.streamed_max,
        Arc::clone(&state),
        deadline,
    );
    send_once(
        generation,
        OutgoingBody {
            body: reqwest::Body::wrap(direct),
            length,
            upload: Some(state),
        },
        lease,
        request_id,
        deadline,
    )
    .await
}

struct OutgoingBody {
    body: reqwest::Body,
    length: u64,
    upload: Option<SharedUploadState>,
}

async fn send_once(
    generation: Arc<HttpGeneration>,
    outgoing: OutgoingBody,
    lease: RequestLease,
    request_id: HeaderValue,
    deadline: tokio::time::Instant,
) -> Result<Response<Body>, HttpFault> {
    let mut url = lease.target().base_url().clone();
    url.set_path(CHAT_PATH);
    url.set_query(None);
    let request = generation
        .client
        .post(url)
        .header(CONTENT_TYPE, canonical_content_type())
        .header(CONTENT_LENGTH, outgoing.length)
        .header(REQUEST_ID_HEADER, request_id)
        .body(outgoing.body);
    let _attempt = authorize_upstream_attempt_at(
        deadline,
        outgoing.upload.as_ref(),
        tokio::time::Instant::now(),
    )?;
    let sent = tokio::select! {
        biased;
        result = request.send() => result,
        () = tokio::time::sleep_until(deadline) => {
            let outcome = deadline_outcome(outgoing.upload.as_ref());
            if outcome.request_probe {
                lease.request_immediate_probe();
            }
            return Err(outcome.fault);
        }
    };
    let response = match sent {
        Ok(response) => response,
        Err(_source) => {
            let fault = selected_send_fault(&outgoing.upload)?;
            if fault == HttpFault::UpstreamProtocolError {
                lease.request_immediate_probe();
            }
            return Err(fault);
        }
    };
    if let Some(fault) = failed_upload(&outgoing.upload)? {
        return Err(fault);
    }
    let response: axum::http::Response<reqwest::Body> = response.into();
    let (parts, body) = response.into_parts();
    let headers = match sanitize_response(parts.status, &parts.headers) {
        Ok(headers) => headers,
        Err(fault) => {
            lease.request_immediate_probe();
            return Err(fault);
        }
    };
    let relay = DirectResponseBody::new(body, lease);
    let mut downstream = Response::new(Body::new(relay));
    *downstream.status_mut() = parts.status;
    *downstream.headers_mut() = headers;
    Ok(downstream)
}

fn failed_upload(upload: &Option<SharedUploadState>) -> Result<Option<HttpFault>, HttpFault> {
    upload
        .as_ref()
        .map(snapshot_upload)
        .transpose()
        .map(|state| {
            state.and_then(|state| match state {
                UploadState::Failed(fault) => Some(fault),
                UploadState::Incomplete | UploadState::Complete => None,
            })
        })
}

fn selected_send_fault(upload: &Option<SharedUploadState>) -> Result<HttpFault, HttpFault> {
    Ok(failed_upload(upload)?.unwrap_or(HttpFault::UpstreamProtocolError))
}

fn deadline_fault(upload: Option<&SharedUploadState>) -> HttpFault {
    match upload.map(snapshot_upload).transpose() {
        Err(fault) => fault,
        Ok(state) => match state {
            Some(UploadState::Incomplete)
            | Some(UploadState::Failed(HttpFault::RequestTimeout)) => HttpFault::RequestTimeout,
            Some(UploadState::Failed(fault)) => fault,
            Some(UploadState::Complete) | None => HttpFault::UpstreamTimeout,
        },
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DeadlineOutcome {
    fault: HttpFault,
    request_probe: bool,
}

fn deadline_outcome(upload: Option<&SharedUploadState>) -> DeadlineOutcome {
    let fault = deadline_fault(upload);
    DeadlineOutcome {
        fault,
        request_probe: fault == HttpFault::UpstreamTimeout,
    }
}

struct UpstreamAttemptAuthorization;

fn authorize_upstream_attempt_at(
    deadline: tokio::time::Instant,
    upload: Option<&SharedUploadState>,
    now: tokio::time::Instant,
) -> Result<UpstreamAttemptAuthorization, HttpFault> {
    check_precommit_deadline_at(deadline, upload, now)?;
    Ok(UpstreamAttemptAuthorization)
}

fn check_precommit_deadline_at(
    deadline: tokio::time::Instant,
    upload: Option<&SharedUploadState>,
    now: tokio::time::Instant,
) -> Result<(), HttpFault> {
    if now >= deadline {
        Err(deadline_fault(upload))
    } else {
        Ok(())
    }
}

fn snapshot_upload(state: &SharedUploadState) -> Result<UploadState, HttpFault> {
    state
        .lock()
        .map(|state| *state)
        .map_err(|_| HttpFault::InternalError)
}

const fn map_admission(error: AdmissionError) -> HttpFault {
    match error {
        AdmissionError::Draining => HttpFault::RouterUnavailable,
        AdmissionError::Overloaded => HttpFault::RouterOverloaded,
        AdmissionError::Internal => HttpFault::InternalError,
    }
}

const fn map_dispatch(error: DispatchError) -> HttpFault {
    match error {
        DispatchError::NoEligibleProfile => HttpFault::NoCompatibleWorker,
        DispatchError::Unavailable | DispatchError::Draining => HttpFault::RouterUnavailable,
        DispatchError::Overloaded => HttpFault::RouterOverloaded,
        DispatchError::Internal => HttpFault::InternalError,
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::cell::Cell;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use super::{
        HttpFault, UploadState, authorize_upstream_attempt_at, deadline_outcome, failed_upload,
        selected_send_fault,
    };

    #[test]
    fn pre_attempt_deadline_gate_is_deterministic() {
        let start = tokio::time::Instant::now();
        let deadline = start + Duration::from_millis(1);
        let attempts = Cell::new(0_u8);
        let result =
            authorize_upstream_attempt_at(deadline, None, deadline).map(|_authorization| {
                attempts.set(attempts.get() + 1);
            });
        assert_eq!(result, Err(HttpFault::UpstreamTimeout));
        assert_eq!(attempts.get(), 0, "an expired request cannot start a send");

        let upload = Arc::new(Mutex::new(UploadState::Incomplete));
        assert!(matches!(
            authorize_upstream_attempt_at(deadline, Some(&upload), deadline),
            Err(HttpFault::RequestTimeout)
        ));
        *upload.lock().expect("update test upload state") = UploadState::Complete;
        assert!(matches!(
            authorize_upstream_attempt_at(deadline, Some(&upload), deadline),
            Err(HttpFault::UpstreamTimeout)
        ));
    }

    #[test]
    fn selected_send_result_is_not_remapped_by_wall_clock_time() {
        assert_eq!(
            selected_send_fault(&None),
            Ok(HttpFault::UpstreamProtocolError)
        );
        let complete = Some(Arc::new(Mutex::new(UploadState::Complete)));
        assert_eq!(
            selected_send_fault(&complete),
            Ok(HttpFault::UpstreamProtocolError)
        );
        let failed = Some(Arc::new(Mutex::new(UploadState::Failed(
            HttpFault::RequestTimeout,
        ))));
        assert_eq!(selected_send_fault(&failed), Ok(HttpFault::RequestTimeout));
    }

    #[test]
    fn completed_upstream_response_accepts_an_incomplete_upload() {
        let incomplete = Some(Arc::new(Mutex::new(UploadState::Incomplete)));

        assert_eq!(failed_upload(&incomplete), Ok(None));
    }

    #[test]
    fn only_worker_owned_precommit_timeouts_request_an_immediate_probe() {
        let complete = Arc::new(Mutex::new(UploadState::Complete));
        let incomplete = Arc::new(Mutex::new(UploadState::Incomplete));
        let client_timeout = Arc::new(Mutex::new(UploadState::Failed(HttpFault::RequestTimeout)));
        let body_fault = Arc::new(Mutex::new(UploadState::Failed(HttpFault::MalformedRequest)));

        for upload in [None, Some(&complete)] {
            let outcome = deadline_outcome(upload);
            assert_eq!(outcome.fault, HttpFault::UpstreamTimeout);
            assert!(outcome.request_probe);
        }
        for (upload, expected) in [
            (Some(&incomplete), HttpFault::RequestTimeout),
            (Some(&client_timeout), HttpFault::RequestTimeout),
            (Some(&body_fault), HttpFault::MalformedRequest),
        ] {
            let outcome = deadline_outcome(upload);
            assert_eq!(outcome.fault, expected);
            assert!(!outcome.request_probe);
        }
    }
}
