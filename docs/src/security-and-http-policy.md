# Security and HTTP Policy

## Vault gate

The application starts behind a master-password gate unless `AMS_SKIP_VAULT=1` is set for development.

`MasterVault` verifies an Argon2id PHC hash from:

- `AMS_MASTER_HASH`, or
- `runs/.master_hash`

The internal `Vault` type encrypts in-memory blobs using:

- Argon2id key material
- HKDF-SHA256 key expansion
- ChaCha20-Poly1305 AEAD

```rust
let key_material = derive_vault_key(master_password, params, &salt)?;
let cipher = ChaCha20Poly1305::new(Key::from_slice(&key_material));
let ciphertext = cipher.encrypt(nonce_ref, Payload { msg: plaintext, aad: VAULT_AAD })?;
```

## Air-gap policy

HTTP safety is enforced by `HttpPolicy`.

```rust
pub struct HttpPolicy {
    pub air_gap_enabled: bool,
    pub allow_local_ollama: bool,
}
```

When `AMS_AIR_GAP=1`:

- non-loopback outbound requests are blocked,
- Ollama is allowed only if host is loopback and `AMS_ALLOW_LOCAL_OLLAMA=1`.

Blocked requests are recorded as transport events in the active run ledger.
