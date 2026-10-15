//! `reglyco login`, `logout` and `whoami`: a GlycoShape login for Level 2
//! glycan ensembles.
//!
//! The login is the device authorization grant of RFC 8628, as `gh auth
//! login` does it: ReGlyco asks the site for a code, the person enters it on
//! the website (signed in there, accepting the Level 2 terms if they have not
//! yet) and approves; ReGlyco then receives a token that is good for Level 2
//! data only, for 90 days or until it is revoked (on the website, or by
//! `reglyco logout`). The site gives Level 2 only to accounts the GlycoShape
//! team has given access.
//!
//! The token is kept in the user's configuration folder
//! (`$REGLYCO_CONFIG_DIR`, else `$XDG_CONFIG_HOME/reglyco`,
//! `~/.config/reglyco`, or `%APPDATA%\reglyco`), in `credentials.json` with
//! mode 0600, per site. It is never taken on the command line, never printed,
//! and sent only to the site it was issued by, over HTTPS (plain HTTP only to
//! this computer, for local development), with `--level 2` requests.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, bail};
use clap::Args;
use reqwest::Url;
use serde::{Deserialize, Serialize};

pub const DEFAULT_API_BASE: &str = "https://glycoshape.org";
const CREDENTIALS: &str = "credentials.json";

#[derive(Debug, Clone, Args)]
pub struct LoginArgs {
    /// GlycoShape site to log in to.
    #[arg(long, default_value = DEFAULT_API_BASE)]
    api_base: String,
    /// How the website lists this login (default: "ReGlyco CLI on <this computer>").
    #[arg(long)]
    name: Option<String>,
}

#[derive(Debug, Clone, Args)]
pub struct SiteArgs {
    /// GlycoShape site.
    #[arg(long, default_value = DEFAULT_API_BASE)]
    api_base: String,
}

pub fn run_login(arguments: LoginArgs) -> anyhow::Result<()> {
    let name = arguments.name.unwrap_or_else(default_client_name);
    login(
        &arguments.api_base,
        &name,
        &config_dir()?,
        &mut std::io::stderr(),
    )
}

pub fn run_logout(arguments: SiteArgs) -> anyhow::Result<()> {
    logout(&arguments.api_base, &config_dir()?, &mut std::io::stderr())
}

pub fn run_whoami(arguments: SiteArgs) -> anyhow::Result<()> {
    whoami(&arguments.api_base, &config_dir()?, &mut std::io::stdout())
}

/// The login to send with Level 2 requests to `api_base`, if there is one
/// for that site (an insecure address never gets one).
pub fn token_for(api_base: &str) -> Option<String> {
    let site = site_key(api_base).ok()?;
    let dir = config_dir().ok()?;
    let login = read_credentials(&dir).ok()?.logins.remove(&site)?;
    (!login.expired()).then_some(login.token)
}

#[derive(Default, Serialize, Deserialize)]
struct Credentials {
    #[serde(default)]
    logins: BTreeMap<String, StoredLogin>,
}

#[derive(Clone, Serialize, Deserialize)]
struct StoredLogin {
    token: String,
    account: String,
    client: String,
    /// Seconds since 1970 when the site ends the login.
    expires_at: u64,
}

impl StoredLogin {
    fn expired(&self) -> bool {
        unix_now() >= self.expires_at
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// The site a login belongs to: the origin of `api_base`, which must be
/// HTTPS (or plain HTTP to this computer, for local development).
fn site_key(api_base: &str) -> anyhow::Result<String> {
    let url =
        Url::parse(api_base.trim()).with_context(|| format!("{api_base} is not a web address"))?;
    let loopback = matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"));
    if !(url.scheme() == "https" || (url.scheme() == "http" && loopback)) {
        bail!("a GlycoShape login is only sent over HTTPS: {api_base} is not an https:// address");
    }
    if url.query().is_some() || url.fragment().is_some() || !matches!(url.path(), "" | "/") {
        bail!("give the site's address only, such as {DEFAULT_API_BASE}");
    }
    Ok(url.origin().ascii_serialization())
}

/// Where the credentials are kept.
fn config_dir() -> anyhow::Result<PathBuf> {
    let from = |name: &str| {
        std::env::var_os(name)
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
    };
    if let Some(dir) = from("REGLYCO_CONFIG_DIR") {
        return Ok(dir);
    }
    if cfg!(windows) {
        if let Some(dir) = from("APPDATA") {
            return Ok(dir.join("reglyco"));
        }
    } else {
        if let Some(dir) = from("XDG_CONFIG_HOME") {
            return Ok(dir.join("reglyco"));
        }
        if let Some(home) = from("HOME") {
            return Ok(home.join(".config").join("reglyco"));
        }
    }
    bail!("no configuration folder for the GlycoShape login: set REGLYCO_CONFIG_DIR")
}

fn read_credentials(dir: &Path) -> anyhow::Result<Credentials> {
    let path = dir.join(CREDENTIALS);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(Credentials::default());
        }
        Err(error) => return Err(error).with_context(|| format!("cannot read {}", path.display())),
    };
    #[cfg(unix)]
    {
        // A file others can read is closed again, as `write_credentials` makes it.
        use std::os::unix::fs::PermissionsExt;
        if let Ok(metadata) = std::fs::metadata(&path)
            && metadata.permissions().mode() & 0o077 != 0
        {
            let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
        }
    }
    serde_json::from_slice(&bytes)
        .with_context(|| format!("{} is not a credentials file", path.display()))
}

/// Write the credentials readable by their owner only, whole or not at all.
fn write_credentials(dir: &Path, credentials: &Credentials) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
    let path = dir.join(CREDENTIALS);
    let partial = dir.join(format!(".{CREDENTIALS}.{}", std::process::id()));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        options.mode(0o600);
        let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700));
    }
    let mut file = options
        .open(&partial)
        .with_context(|| format!("cannot write {}", partial.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    file.write_all(&serde_json::to_vec_pretty(credentials)?)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&partial, &path).with_context(|| format!("cannot write {}", path.display()))?;
    Ok(())
}

fn default_client_name() -> String {
    let host = std::env::var("HOSTNAME")
        .ok()
        .or_else(|| std::fs::read_to_string("/etc/hostname").ok())
        .or_else(|| std::env::var("COMPUTERNAME").ok())
        .map(|host| {
            host.trim()
                .chars()
                .filter(|c| c.is_ascii_graphic())
                .take(60)
                .collect::<String>()
        })
        .filter(|host| !host.is_empty());
    match host {
        Some(host) => format!("ReGlyco CLI on {host}"),
        None => "ReGlyco CLI".into(),
    }
}

fn client() -> anyhow::Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder()
        .user_agent(concat!("reglyco/", env!("CARGO_PKG_VERSION")))
        .timeout(Duration::from_secs(30))
        // The token goes to the site it was issued by, not wherever that sends us.
        .redirect(reqwest::redirect::Policy::none())
        .build()?)
}

#[derive(Debug, Deserialize)]
struct DeviceCode {
    device_code: String,
    user_code: String,
    verification_uri: String,
    verification_uri_complete: Option<String>,
    expires_in: u64,
    interval: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct TokenAnswer {
    access_token: Option<String>,
    expires_in: Option<u64>,
    user: Option<Named>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Named {
    name: String,
}

fn login(api_base: &str, name: &str, dir: &Path, out: &mut dyn Write) -> anyhow::Result<()> {
    let site = site_key(api_base)?;
    let http = client()?;
    let response = http
        .post(format!("{site}/api/account/device/code"))
        .json(&serde_json::json!({ "client_name": name, "scope": "level2" }))
        .send()
        .with_context(|| format!("cannot reach {site}"))?;
    if response.status().as_u16() == 404 {
        bail!("{site} has no GlycoShape logins");
    }
    let code: DeviceCode = response
        .error_for_status()
        .with_context(|| format!("{site} did not start a login"))?
        .json()?;
    writeln!(out, "To log ReGlyco in to {site} for Level 2 data, open")?;
    writeln!(out, "  {}", code.verification_uri)?;
    writeln!(out, "and enter the code  {}", code.user_code)?;
    if let Some(complete) = &code.verification_uri_complete {
        writeln!(out, "(or open {complete})")?;
    }
    writeln!(out, "Waiting for the approval on the website...")?;
    let deadline = Instant::now() + Duration::from_secs(code.expires_in.max(1));
    let mut interval = code.interval.unwrap_or(5);
    let answer = loop {
        std::thread::sleep(Duration::from_secs(interval));
        if Instant::now() >= deadline {
            bail!("the code expired before it was approved: run `reglyco login` again");
        }
        let answer: TokenAnswer = http
            .post(format!("{site}/api/account/device/token"))
            .json(&serde_json::json!({
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "device_code": code.device_code,
            }))
            .send()
            .with_context(|| format!("cannot reach {site}"))?
            .json()
            .with_context(|| format!("{site} answered the login with something else"))?;
        match answer.error.as_deref() {
            None if answer.access_token.is_some() => break answer,
            Some("authorization_pending") => {}
            Some("slow_down") => interval += 5,
            Some("access_denied") => bail!(
                "the login was denied on the website (or your account has no Level 2 access: ask the GlycoShape team, support@glycoshape.org)"
            ),
            Some("expired_token") => {
                bail!("the code expired before it was approved: run `reglyco login` again")
            }
            Some(other) => bail!("{site} refused the login ({other})"),
            None => bail!("{site} answered the login without a token"),
        }
    };
    let account = answer.user.map(|user| user.name).unwrap_or_default();
    let mut credentials = read_credentials(dir)?;
    credentials.logins.insert(
        site.clone(),
        StoredLogin {
            token: answer.access_token.expect("checked above"),
            account: account.clone(),
            client: name.into(),
            expires_at: unix_now() + answer.expires_in.unwrap_or(90 * 24 * 3600),
        },
    );
    write_credentials(dir, &credentials)?;
    writeln!(
        out,
        "Logged in to {site} as {account}; `--level 2` now uses this login."
    )?;
    Ok(())
}

fn logout(api_base: &str, dir: &Path, out: &mut dyn Write) -> anyhow::Result<()> {
    let site = site_key(api_base)?;
    let mut credentials = read_credentials(dir)?;
    let Some(login) = credentials.logins.remove(&site) else {
        writeln!(out, "Not logged in to {site}.")?;
        return Ok(());
    };
    // End the login on the site too; it may have ended there already.
    let ended = client()?
        .delete(format!("{site}/api/account/level2/token"))
        .bearer_auth(&login.token)
        .send()
        .map(|response| response.status().is_success() || response.status().as_u16() == 401);
    write_credentials(dir, &credentials)?;
    match ended {
        Ok(true) => writeln!(out, "Logged out of {site}.")?,
        _ => writeln!(
            out,
            "Removed the login from this computer, but {site} could not be told: end it on {site}/account#level-2."
        )?,
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Whoami {
    user: WhoamiUser,
    token: WhoamiToken,
    granted: bool,
    terms_accepted: bool,
    terms_url: String,
}

#[derive(Debug, Deserialize)]
struct WhoamiUser {
    name: String,
    orcid: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WhoamiToken {
    name: String,
    expires_at: String,
}

fn whoami(api_base: &str, dir: &Path, out: &mut dyn Write) -> anyhow::Result<()> {
    let site = site_key(api_base)?;
    let mut credentials = read_credentials(dir)?;
    let Some(login) = credentials.logins.get(&site).cloned() else {
        bail!("not logged in to {site}: run `reglyco login`");
    };
    let response = client()?
        .get(format!("{site}/api/account/level2/whoami"))
        .bearer_auth(&login.token)
        .send()
        .with_context(|| format!("cannot reach {site}"))?;
    if response.status().as_u16() == 401 || login.expired() {
        credentials.logins.remove(&site);
        write_credentials(dir, &credentials)?;
        bail!("your login to {site} has expired or was revoked: run `reglyco login` again");
    }
    let who: Whoami = response
        .error_for_status()
        .with_context(|| format!("{site} did not say who this login is"))?
        .json()?;
    let orcid = who
        .user
        .orcid
        .map(|orcid| format!(" (ORCID {orcid})"))
        .unwrap_or_default();
    writeln!(out, "Logged in to {site} as {}{orcid}.", who.user.name)?;
    writeln!(
        out,
        "This login: \"{}\", until {}.",
        who.token.name,
        who.token
            .expires_at
            .get(..10)
            .unwrap_or(&who.token.expires_at)
    )?;
    if !who.granted {
        writeln!(
            out,
            "Level 2: not available. Your account does not have Level 2 access: ask the GlycoShape team for Level 2 access (support@glycoshape.org)."
        )?;
    } else if !who.terms_accepted {
        writeln!(
            out,
            "Level 2: waiting for the terms. Accept the current Level 2 terms at {}",
            who.terms_url
        )?;
    } else {
        writeln!(out, "Level 2: available (`--level 2`).")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    #[derive(Debug, Clone)]
    struct Asked {
        method: String,
        path: String,
        authorization: Option<String>,
        body: String,
    }

    /// A site that answers each request with the next of `responses`
    /// (status and JSON) and remembers what it was asked.
    fn site(
        responses: Vec<(u16, String)>,
    ) -> (String, Arc<Mutex<Vec<Asked>>>, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
        let asked = Arc::new(Mutex::new(Vec::new()));
        let record = asked.clone();
        let server = std::thread::spawn(move || {
            for (status, body) in responses {
                let (stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream);
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                let mut parts = line.split_whitespace();
                let (method, path) = (
                    parts.next().unwrap().to_string(),
                    parts.next().unwrap().to_string(),
                );
                let (mut length, mut authorization) = (0, None);
                loop {
                    let mut header = String::new();
                    reader.read_line(&mut header).unwrap();
                    let header = header.trim_end();
                    if header.is_empty() {
                        break;
                    }
                    let (name, value) = header.split_once(':').unwrap();
                    match name.to_ascii_lowercase().as_str() {
                        "content-length" => length = value.trim().parse().unwrap(),
                        "authorization" => authorization = Some(value.trim().to_string()),
                        _ => {}
                    }
                }
                let mut body_bytes = vec![0; length];
                reader.read_exact(&mut body_bytes).unwrap();
                record.lock().unwrap().push(Asked {
                    method,
                    path,
                    authorization,
                    body: String::from_utf8(body_bytes).unwrap(),
                });
                let mut stream = reader.into_inner();
                write!(
                    stream,
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .unwrap();
            }
        });
        (base, asked, server)
    }

    fn config() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "reglyco-login-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        dir.join("reglyco")
    }

    const TOKEN: &str = "gs2_0123456789abcdefghijklmnopqrstuvwxyzABCDEFG";

    fn device_code(interval: u64) -> String {
        format!(
            r#"{{"device_code":"dc","user_code":"WDJB-MJHT","verification_uri":"https://glycoshape.example/device","verification_uri_complete":"https://glycoshape.example/device?code=WDJB-MJHT","expires_in":900,"interval":{interval}}}"#
        )
    }

    #[test]
    fn a_login_is_only_for_an_https_site_and_kept_per_site() {
        assert_eq!(
            site_key("https://glycoshape.org").unwrap(),
            "https://glycoshape.org"
        );
        assert_eq!(
            site_key("https://glycoshape.org/").unwrap(),
            "https://glycoshape.org"
        );
        assert_eq!(
            site_key("http://127.0.0.1:8080").unwrap(),
            "http://127.0.0.1:8080"
        );
        assert_eq!(
            site_key("http://localhost:5173/").unwrap(),
            "http://localhost:5173"
        );
        for refused in [
            "http://glycoshape.org",
            "ftp://glycoshape.org",
            "https://glycoshape.org/api",
            "https://glycoshape.org/?token=1",
            "glycoshape.org",
        ] {
            assert!(site_key(refused).is_err(), "{refused}");
        }
    }

    #[test]
    fn logging_in_waits_for_the_approval_and_keeps_the_token_to_its_owner() {
        let (base, asked, server) = site(vec![
            (200, device_code(0)),
            (400, r#"{"error":"authorization_pending"}"#.into()),
            (400, r#"{"error":"authorization_pending"}"#.into()),
            (
                200,
                format!(r#"{{"access_token":"{TOKEN}","token_type":"Bearer","expires_in":7776000,"scope":"level2","user":{{"name":"Cy"}}}}"#),
            ),
            (
                200,
                r#"{"user":{"name":"Cy","orcid":"0000-0002-1825-0097"},"token":{"id":"l2t_1","name":"ReGlyco CLI on lab-pc","createdAt":"2026-10-15T10:00:00Z","expiresAt":"2027-01-13T10:00:00Z","lastUsedAt":null},"granted":true,"termsVersion":"v","termsAccepted":true,"termsUrl":"https://glycoshape.example/account#level-2"}"#.into(),
            ),
            (200, r#"{"ok":true}"#.into()),
        ]);
        let dir = config();
        let mut out = Vec::new();
        login(&base, "ReGlyco CLI on lab-pc", &dir, &mut out).unwrap();
        let printed = String::from_utf8(out).unwrap();
        assert!(printed.contains("https://glycoshape.example/device"));
        assert!(printed.contains("WDJB-MJHT"));
        assert!(printed.contains("as Cy"));
        assert!(!printed.contains(TOKEN), "the token is never printed");

        // Kept for this site only, readable by its owner only.
        let file = dir.join(CREDENTIALS);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&file).unwrap().permissions().mode() & 0o777,
                0o600
            );
            assert_eq!(
                std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777,
                0o700
            );
        }
        let stored = read_credentials(&dir).unwrap();
        assert_eq!(stored.logins[&base].token, TOKEN);
        assert_eq!(stored.logins[&base].account, "Cy");
        assert!(!stored.logins.contains_key("https://glycoshape.org"));

        let mut out = Vec::new();
        whoami(&base, &dir, &mut out).unwrap();
        let printed = String::from_utf8(out).unwrap();
        assert!(
            printed.contains("as Cy (ORCID 0000-0002-1825-0097)"),
            "{printed}"
        );
        assert!(printed.contains("Level 2: available"));
        assert!(printed.contains("until 2027-01-13"));

        let mut out = Vec::new();
        logout(&base, &dir, &mut out).unwrap();
        assert!(String::from_utf8(out).unwrap().contains("Logged out"));
        assert!(read_credentials(&dir).unwrap().logins.is_empty());
        server.join().unwrap();

        let asked = asked.lock().unwrap();
        assert_eq!(asked[0].method, "POST");
        assert_eq!(asked[0].path, "/api/account/device/code");
        assert!(
            asked[0]
                .body
                .contains(r#""client_name":"ReGlyco CLI on lab-pc""#)
        );
        assert!(asked[1].body.contains(r#""device_code":"dc""#));
        assert!(
            asked[1]
                .body
                .contains("urn:ietf:params:oauth:grant-type:device_code")
        );
        assert!(
            asked[..4]
                .iter()
                .all(|request| request.authorization.is_none())
        );
        assert_eq!(asked[4].path, "/api/account/level2/whoami");
        assert_eq!(
            asked[4].authorization.as_deref(),
            Some(format!("Bearer {TOKEN}").as_str())
        );
        assert_eq!(
            (asked[5].method.as_str(), asked[5].path.as_str()),
            ("DELETE", "/api/account/level2/token")
        );
        assert_eq!(
            asked[5].authorization.as_deref(),
            Some(format!("Bearer {TOKEN}").as_str())
        );
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    #[test]
    fn a_denied_login_and_an_account_without_access_are_explained() {
        let (base, _, server) = site(vec![
            (200, device_code(0)),
            (400, r#"{"error":"access_denied"}"#.into()),
        ]);
        let dir = config();
        let error = login(&base, "ReGlyco CLI", &dir, &mut Vec::new())
            .unwrap_err()
            .to_string();
        server.join().unwrap();
        assert!(error.contains("denied"), "{error}");
        assert!(error.contains("ask the GlycoShape team"), "{error}");
        assert!(!dir.join(CREDENTIALS).exists());

        // Logged in, but the terms changed, or the access was taken away.
        let mut credentials = Credentials::default();
        credentials.logins.insert(
            "http://127.0.0.1:1".into(),
            StoredLogin {
                token: TOKEN.into(),
                account: "Cy".into(),
                client: "c".into(),
                expires_at: unix_now() + 60,
            },
        );
        for (granted, accepted, expected) in [
            (
                true,
                false,
                "Accept the current Level 2 terms at https://glycoshape.example/account#level-2",
            ),
            (false, false, "ask the GlycoShape team for Level 2 access"),
        ] {
            let body = format!(
                r#"{{"user":{{"name":"Cy","orcid":null}},"token":{{"name":"c","expiresAt":"2027-01-13T10:00:00Z"}},"granted":{granted},"termsAccepted":{accepted},"termsUrl":"https://glycoshape.example/account#level-2"}}"#
            );
            let (base, _, server) = site(vec![(200, body)]);
            let mut stored = Credentials::default();
            stored.logins.insert(
                base.clone(),
                credentials.logins["http://127.0.0.1:1"].clone(),
            );
            write_credentials(&dir, &stored).unwrap();
            let mut out = Vec::new();
            whoami(&base, &dir, &mut out).unwrap();
            server.join().unwrap();
            assert!(String::from_utf8(out).unwrap().contains(expected));
        }

        // A login the site ended is forgotten here too.
        let (base, _, server) = site(vec![(401, r#"{"code":"level2_token_invalid"}"#.into())]);
        let mut stored = Credentials::default();
        stored.logins.insert(
            base.clone(),
            credentials.logins["http://127.0.0.1:1"].clone(),
        );
        write_credentials(&dir, &stored).unwrap();
        let error = whoami(&base, &dir, &mut Vec::new())
            .unwrap_err()
            .to_string();
        server.join().unwrap();
        assert!(error.contains("run `reglyco login` again"), "{error}");
        assert!(read_credentials(&dir).unwrap().logins.is_empty());
        assert!(
            whoami(&base, &dir, &mut Vec::new())
                .unwrap_err()
                .to_string()
                .contains("not logged in")
        );
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    #[test]
    fn slow_down_makes_the_client_wait_longer() {
        // RFC 8628: every `slow_down` adds five seconds to the interval.
        let (base, asked, server) = site(vec![
            (200, device_code(0)),
            (400, r#"{"error":"slow_down"}"#.into()),
            (400, r#"{"error":"expired_token"}"#.into()),
        ]);
        let started = Instant::now();
        let error = login(&base, "c", &config(), &mut Vec::new())
            .unwrap_err()
            .to_string();
        server.join().unwrap();
        assert!(error.contains("expired"), "{error}");
        assert!(started.elapsed() >= Duration::from_secs(5));
        assert_eq!(asked.lock().unwrap().len(), 3);
    }
}
