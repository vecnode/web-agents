//! Writes a PHC Argon2 hash to `runs/.master_hash` (same params as `MasterVault` verification).

use argon2::Argon2;
use argon2::password_hash::{PasswordHasher, SaltString, rand_core::OsRng};
use std::io::Write;
use std::path::Path;

fn main() -> std::io::Result<()> {
    let path = Path::new("runs").join(".master_hash");
    println!("This writes a one-line Argon2 hash to {}.", path.display());
    if path.exists() {
        eprint!("File already exists. Overwrite? [y/N]: ");
        std::io::stderr().flush()?;
        let mut line = String::new();
        std::io::stdin().read_line(&mut line)?;
        if !line.trim().eq_ignore_ascii_case("y") {
            println!("Aborted.");
            return Ok(());
        }
    }

    let pw = rpassword::prompt_password("Master password: ")
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    let confirm = rpassword::prompt_password("Again: ")
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    if pw != confirm {
        eprintln!("Passwords do not match.");
        std::process::exit(1);
    }

    let salt = SaltString::generate(&mut OsRng);
    let phc = Argon2::default()
        .hash_password(pw.as_bytes(), &salt)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?
        .to_string();

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, format!("{phc}\n"))?;

    println!(
        "Wrote {} (keep runs/ out of version control; do not commit the password).",
        path.display()
    );
    Ok(())
}
