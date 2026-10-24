//! The JSON-RPC 2.0 envelope and the HTTP server behind `jackdaw serve`.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/core/server.lua`: one client at a time, HTTP/1.1, `POST /` only, a 64 KiB body
//! limit, `Connection: close` after every response.
//!
//! - **Envelope** ([`handle_jsonrpc`]): the body must decode to a table with `jsonrpc == "2.0"`
//!   and an integer or string `id`; errors before the `id` is accepted leave `id` out of the
//!   response (Lua's `nil`). Success is `{jsonrpc, result, id}`; an error is
//!   `{jsonrpc, error = {code, message, data = {name}}, id}`, HTTP 200 either way.
//! - **HTTP errors** (bad request line, wrong method or path, body too large) answer 400/405/404
//!   with a `BAD_REQUEST` error whose `id` is the *previous* request's (`current_request_id` is
//!   only reset by the envelope checks).
//! - **Encoding failures**: where the live encoder raises (NaN or infinity in the result) and
//!   BalatroBot sends nothing, the sim answers `INTERNAL_ERROR` with
//!   `Failed to encode response: <encoder error>`.
//!
//! The server needs no HTTP library: requests are parsed exactly as `try_parse_http` does.
//! Its one addition is a `Server: jackdaw` header, so a client can tell the simulator from the
//! live game (`LiveBackend` waits before a live `cash_out`, `docs/balatrobot.md`).

use std::io::{Read, Write};
use std::net::{IpAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::time::Duration;

use crate::errors::{BotError, ErrorName};
use crate::json::{self, Json};
use crate::lua::{self, Value};
use crate::session::Session;

/// `MAX_BODY_SIZE`.
pub const MAX_BODY_SIZE: usize = 65_536;

/// What the envelope keeps between requests (`BB_SERVER.current_request_id`).
#[derive(Clone, Debug, Default)]
pub struct Envelope {
    pub current_request_id: Option<Value>,
}

fn wrap_error(e: &BotError, id: Option<&Value>) -> Json {
    let mut entries = vec![
        ("jsonrpc".to_owned(), Json::str("2.0")),
        (
            "error".to_owned(),
            Json::obj([
                ("code", Json::Num(e.name.code() as f64)),
                ("message", Json::str(e.message.clone())),
                ("data", Json::obj([("name", Json::str(e.name.name()))])),
            ]),
        ),
    ];
    if let Some(id) = id.and_then(|v| json::from_value(v).ok()) {
        entries.push(("id".to_owned(), id));
    }
    Json::Obj(entries)
}

fn wrap_result(result: Json, id: Option<&Value>) -> Json {
    let mut entries = vec![
        ("jsonrpc".to_owned(), Json::str("2.0")),
        ("result".to_owned(), result),
    ];
    if let Some(id) = id.and_then(|v| json::from_value(v).ok()) {
        entries.push(("id".to_owned(), id));
    }
    Json::Obj(entries)
}

/// `BB_SERVER.send_response(response)`: the wrapped response, encoded.
fn send(response: Result<Json, BotError>, id: Option<&Value>) -> String {
    let wrapped = match response {
        Ok(r) => wrap_result(r, id),
        Err(e) => wrap_error(&e, id),
    };
    match json::encode(&wrapped) {
        Ok(text) => text,
        Err(e) => {
            let fallback = BotError::internal(format!("Failed to encode response: {e}"));
            json::encode(&wrap_error(&fallback, id))
                .or_else(|_| json::encode(&wrap_error(&fallback, None)))
                .expect("an error envelope encodes")
        }
    }
}

/// `handle_jsonrpc(body, dispatcher)` (server.lua:290-352) on a session.
pub fn handle_jsonrpc(session: &mut Session, body: &[u8]) -> String {
    let mut envelope = std::mem::take(&mut session.envelope);
    let out = handle_with(session, &mut envelope, body);
    session.envelope = envelope;
    out
}

/// [`handle_jsonrpc`] with an explicit envelope state.
pub fn handle_with(session: &mut Session, envelope: &mut Envelope, body: &[u8]) -> String {
    let parsed = match json::decode(body) {
        Ok(Some(Value::Table(t))) => t,
        _ => {
            envelope.current_request_id = None;
            return send(
                Err(BotError::bad_request("Invalid JSON in request body")),
                None,
            );
        }
    };
    if parsed.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        envelope.current_request_id = parsed.get("id").cloned();
        return send(
            Err(BotError::bad_request(
                "Invalid JSON-RPC version: expected '2.0'",
            )),
            envelope.current_request_id.as_ref(),
        );
    }
    let id = parsed.get("id").cloned();
    let refuse = |envelope: &mut Envelope, msg: &str| {
        envelope.current_request_id = None;
        send(Err(BotError::bad_request(msg)), None)
    };
    match &id {
        None => return refuse(envelope, "Invalid Request: 'id' field is required"),
        Some(Value::Num(x)) if x.floor() != *x => {
            return refuse(
                envelope,
                "Invalid Request: 'id' must be an integer, not a float",
            );
        }
        Some(Value::Num(_) | Value::Str(_)) => {}
        Some(_) => {
            return refuse(
                envelope,
                "Invalid Request: 'id' must be an integer or string",
            );
        }
    }
    envelope.current_request_id = id.clone();
    let method = parsed.get("method").and_then(Value::as_str);
    let response = crate::dispatcher::dispatch(session, method, parsed.get("params"), false);
    send(response, id.as_ref())
}

/// The JSON body of an HTTP-level error (`send_http_error`, server.lua:262-285).
fn http_error_body(status: u16, message: &str, envelope: &Envelope) -> String {
    let name = if status == 500 {
        ErrorName::InternalError
    } else {
        ErrorName::BadRequest
    };
    send(
        Err(BotError::new(name, message)),
        envelope.current_request_id.as_ref(),
    )
}

/// `format_http_response(status_code, status_text, body)`.
fn http_response(status: u16, body: &str) -> Vec<u8> {
    let text = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        _ => "Error",
    };
    format!(
        "HTTP/1.1 {status} {text}\r\nServer: jackdaw\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .into_bytes()
}

/// A parsed HTTP request, or the error `try_parse_http` reports.
enum Parsed {
    Incomplete,
    Error(&'static str),
    Request {
        method: String,
        path: String,
        body: Vec<u8>,
    },
}

/// `try_parse_http()` (server.lua:181-241) on the bytes received so far.
fn try_parse_http(buffer: &[u8]) -> Parsed {
    let Some(header_end) = buffer.windows(4).position(|w| w == b"\r\n\r\n") else {
        return Parsed::Incomplete;
    };
    let header = String::from_utf8_lossy(&buffer[..header_end]);
    let lines: Vec<&str> = header
        .split(['\r', '\n'])
        .filter(|l| !l.is_empty())
        .collect();
    let Some(first) = lines.first() else {
        return Parsed::Error("Empty request");
    };
    // `^(%u+)%s+(%S+)%s+HTTP/(%d%.%d)`.
    let mut parts = first.split_ascii_whitespace();
    let (Some(method), Some(path), Some(version)) = (parts.next(), parts.next(), parts.next())
    else {
        return Parsed::Error("Invalid request line");
    };
    let version_ok = version
        .strip_prefix("HTTP/")
        .map(str::as_bytes)
        .is_some_and(|v| {
            v.len() >= 3 && v[0].is_ascii_digit() && v[1] == b'.' && v[2].is_ascii_digit()
        });
    if method.is_empty() || !method.bytes().all(|b| b.is_ascii_uppercase()) || !version_ok {
        return Parsed::Error("Invalid request line");
    }
    let mut content_length = 0usize;
    for line in &lines[1..] {
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            // `tonumber(...)`; an unreadable length counts as none.
            content_length = lua::tonumber(value.trim())
                .filter(|n| *n >= 0.0)
                .map_or(0, |n| n as usize);
        }
    }
    let body_start = header_end + 4;
    let body = if method == "POST" {
        if content_length > MAX_BODY_SIZE {
            return Parsed::Error("Request body too large");
        }
        if buffer.len() - body_start < content_length {
            return Parsed::Incomplete;
        }
        buffer[body_start..body_start + content_length].to_vec()
    } else {
        Vec::new()
    };
    Parsed::Request {
        method: method.to_owned(),
        path: path.to_owned(),
        body,
    }
}

/// Serve one connection: read one request, answer, close.
fn serve_connection(stream: &mut TcpStream, session: &mut Session) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 8192];
    let (status, body) = loop {
        match try_parse_http(&buffer) {
            Parsed::Incomplete => {}
            Parsed::Error(msg) => break (400, http_error_body(400, msg, &session.envelope)),
            Parsed::Request { method, path, body } => {
                if method != "POST" {
                    break (
                        405,
                        http_error_body(
                            405,
                            "Method not allowed. Use POST for JSON-RPC requests",
                            &session.envelope,
                        ),
                    );
                }
                if path != "/" {
                    break (
                        404,
                        http_error_body(
                            404,
                            "Not found. Use POST to '/' for JSON-RPC requests",
                            &session.envelope,
                        ),
                    );
                }
                break (200, session.handle_body(&body));
            }
        }
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            // The client closed before sending a whole request.
            return Ok(());
        }
        buffer.extend_from_slice(&chunk[..n]);
    };
    stream.write_all(&http_response(status, &body))?;
    stream.flush()?;
    if status != 200 {
        // A refused request may still be arriving or unread (a body over the limit, or a PUT's
        // body, which a 405 never reads): half-close and drain it, so closing doesn't reset the
        // connection before the client reads the response (Windows sends RST when unread data
        // is discarded). server.lua closes at once; the response is the same.
        let _ = stream.shutdown(std::net::Shutdown::Write);
        let _ = stream.set_read_timeout(Some(Duration::from_millis(250)));
        let mut drained = 0usize;
        while drained < 4 * MAX_BODY_SIZE {
            match stream.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => drained += n,
            }
        }
    }
    let _ = stream.shutdown(std::net::Shutdown::Both);
    Ok(())
}

/// Whether `host` only reaches this machine.
pub fn is_loopback(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    match host.trim_matches(['[', ']']).parse::<IpAddr>() {
        Ok(ip) => ip.is_loopback(),
        Err(_) => false,
    }
}

/// A bound server, ready to [`Server::run`].
pub struct Server {
    listener: TcpListener,
    pub session: Session,
}

impl Server {
    /// Bind `host:port` (`port` 0 picks a free port). A non-loopback host needs a filesystem
    /// root for `save` and `load`.
    pub fn bind(host: &str, port: u16, session: Session) -> Result<Server, String> {
        if !is_loopback(host) && session.config.fs_root.is_none() {
            return Err(format!(
                "binding to the non-loopback address {host} needs a filesystem root for save/load (--fs-root)"
            ));
        }
        let addr = (host.trim_matches(['[', ']']), port)
            .to_socket_addrs()
            .map_err(|e| format!("bad address {host}:{port}: {e}"))?
            .next()
            .ok_or_else(|| format!("no address for {host}:{port}"))?;
        let listener =
            TcpListener::bind(addr).map_err(|e| format!("Failed to bind to port {port}: {e}"))?;
        Ok(Server { listener, session })
    }

    /// The bound port.
    pub fn port(&self) -> u16 {
        self.listener.local_addr().map_or(0, |a| a.port())
    }

    /// Accept and answer connections one at a time until `stop()` returns true (checked about
    /// every 100 ms while idle).
    pub fn run(&mut self, mut stop: impl FnMut() -> bool) -> std::io::Result<()> {
        self.listener.set_nonblocking(true)?;
        loop {
            if stop() {
                return Ok(());
            }
            match self.listener.accept() {
                Ok((mut stream, _)) => {
                    stream.set_nonblocking(false)?;
                    // A broken connection only loses its own response.
                    let _ = serve_connection(&mut stream, &mut self.session);
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(e) => return Err(e),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(s: &mut Session, body: &str) -> serde_json::Value {
        serde_json::from_str(&s.handle_body(body.as_bytes())).unwrap()
    }

    #[test]
    fn envelope_rules() {
        let mut s = Session::default();
        let r = call(&mut s, "{invalid json}");
        assert_eq!(r["error"]["data"]["name"], "BAD_REQUEST");
        assert!(r.get("id").is_none(), "no id before the id check");
        let r = call(&mut s, r#"{"jsonrpc": "1.0", "method": "health", "id": 7}"#);
        assert_eq!(r["id"], 7);
        assert_eq!(
            r["error"]["message"],
            "Invalid JSON-RPC version: expected '2.0'"
        );
        let r = call(
            &mut s,
            r#"{"jsonrpc": "2.0", "method": "health", "id": 1.5}"#,
        );
        assert_eq!(
            r["error"]["message"],
            "Invalid Request: 'id' must be an integer, not a float"
        );
        let r = call(
            &mut s,
            r#"{"jsonrpc": "2.0", "method": "health", "id": [1]}"#,
        );
        assert_eq!(
            r["error"]["message"],
            "Invalid Request: 'id' must be an integer or string"
        );
        let r = call(
            &mut s,
            r#"{"jsonrpc": "2.0", "method": "health", "id": null}"#,
        );
        assert_eq!(
            r["error"]["message"],
            "Invalid Request: 'id' field is required"
        );
        let r = call(
            &mut s,
            r#"{"jsonrpc": "2.0", "method": "health", "id": "x"}"#,
        );
        assert_eq!(r["result"]["status"], "ok");
        assert_eq!(r["id"], "x");
        let r = call(&mut s, r#"[1, 2]"#);
        assert_eq!(
            r["error"]["message"],
            "Invalid JSON-RPC version: expected '2.0'"
        );
    }

    #[test]
    fn http_parsing() {
        assert!(matches!(
            try_parse_http(b"POST / HTTP/1.1\r\n"),
            Parsed::Incomplete
        ));
        assert!(matches!(
            try_parse_http(b"\r\n\r\n"),
            Parsed::Error("Empty request")
        ));
        assert!(matches!(
            try_parse_http(b"post / HTTP/1.1\r\n\r\n"),
            Parsed::Error("Invalid request line")
        ));
        assert!(matches!(
            try_parse_http(b"POST / HTTP/1.1\r\nContent-Length: 5\r\n\r\nab"),
            Parsed::Incomplete
        ));
        let big = format!(
            "POST / HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
            MAX_BODY_SIZE + 1
        );
        assert!(matches!(
            try_parse_http(big.as_bytes()),
            Parsed::Error("Request body too large")
        ));
        match try_parse_http(b"GET /x HTTP/1.1\r\nHost: a\r\n\r\n") {
            Parsed::Request { method, path, body } => {
                assert_eq!((method.as_str(), path.as_str()), ("GET", "/x"));
                assert!(body.is_empty());
            }
            _ => panic!("parses"),
        }
        assert!(is_loopback("127.0.0.1") && is_loopback("localhost") && is_loopback("::1"));
        assert!(!is_loopback("0.0.0.0"));
    }
}
