use crate::http_client::get_outgoing_http_log_lines;
use rocket::figment::Figment;
use rocket::serde::json::Json;
use rocket::{Build, Config, Rocket, routes};
use serde::Serialize;
use tokio::runtime::Handle;

#[derive(Clone, Debug)]
struct WebConfig {
    enabled: bool,
    address: String,
    port: u16,
}

impl WebConfig {
    fn from_env() -> Self {
        let enabled = parse_bool_env("AMS_WEB_ENABLED", false);
        let address = std::env::var("AMS_WEB_ADDRESS").unwrap_or_else(|_| "127.0.0.1".to_string());
        let port = std::env::var("AMS_WEB_PORT")
            .ok()
            .and_then(|v| v.parse::<u16>().ok())
            .unwrap_or(8000);

        Self {
            enabled,
            address,
            port,
        }
    }
}

fn parse_bool_env(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => default,
        },
        Err(_) => default,
    }
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    service: &'static str,
}

#[derive(Serialize)]
struct OutgoingHttpLogResponse {
    lines: Vec<String>,
}

#[rocket::get("/health")]
fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        service: "ams-agents",
    })
}

#[rocket::get("/outgoing-http-log")]
fn outgoing_http_log() -> Json<OutgoingHttpLogResponse> {
    Json(OutgoingHttpLogResponse {
        lines: get_outgoing_http_log_lines(),
    })
}

fn build_rocket(config: &WebConfig) -> Rocket<Build> {
    let figment: Figment = Config::figment()
        .merge(("address", config.address.clone()))
        .merge(("port", config.port));

    rocket::custom(figment).mount("/api", routes![health, outgoing_http_log])
}

pub fn start_embedded_server_if_enabled(rt_handle: &Handle) -> bool {
    let config = WebConfig::from_env();
    if !config.enabled {
        return false;
    }

    rt_handle.spawn(async move {
        if let Err(err) = build_rocket(&config).launch().await {
            eprintln!("Embedded Rocket server failed: {err}");
        }
    });

    true
}

#[cfg(test)]
mod tests {
    use super::parse_bool_env;

    #[test]
    fn parse_bool_env_honors_default_when_unset() {
        let key = "AMS_TEST_WEB_BOOL_MISSING";
        unsafe { std::env::remove_var(key) };
        assert!(!parse_bool_env(key, false));
        assert!(parse_bool_env(key, true));
    }

    #[test]
    fn parse_bool_env_supports_truthy_values() {
        let key = "AMS_TEST_WEB_BOOL_TRUE";
        unsafe { std::env::set_var(key, "yes") };
        assert!(parse_bool_env(key, false));
        unsafe { std::env::remove_var(key) };
    }

    #[test]
    fn parse_bool_env_supports_falsey_values() {
        let key = "AMS_TEST_WEB_BOOL_FALSE";
        unsafe { std::env::set_var(key, "off") };
        assert!(!parse_bool_env(key, true));
        unsafe { std::env::remove_var(key) };
    }
}
