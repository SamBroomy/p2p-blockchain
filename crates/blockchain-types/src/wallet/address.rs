use core::fmt;

use ed25519_dalek::{Signature, SignatureError, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};

/// Custom address type wrapping `VerifyingKey` for better display and future abstraction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Address(VerifyingKey);

impl Address {
    /// Create an Address from a `VerifyingKey`
    pub fn from_verifying_key(key: VerifyingKey) -> Self {
        Self(key)
    }

    /// Get the underlying `VerifyingKey`
    pub fn as_verifying_key(&self) -> &VerifyingKey {
        &self.0
    }

    /// Get the raw bytes of the address
    pub fn as_bytes(&self) -> &[u8; 32] {
        self.0.as_bytes()
    }

    /// Verify a signature for given message bytes
    pub fn verify(
        &self,
        message: impl AsRef<[u8]>,
        signature: &Signature,
    ) -> Result<(), AddressError> {
        self.0
            .verify(message.as_ref(), signature)
            .map_err(AddressError::SignatureError)
    }

    /// Create from raw bytes
    pub fn from_bytes(bytes: &[u8; 32]) -> Result<Self, AddressError> {
        VerifyingKey::from_bytes(bytes)
            .map(Self)
            .map_err(AddressError::SignatureError)
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display first 8 bytes as hex (like Bitcoin/Ethereum addresses)
        let bytes = self.0.as_bytes();
        write!(f, "{}", hex::encode(&bytes[..8]))
    }
}

impl Serialize for Address {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Serialize as hex string for readability
        let hex_string = hex::encode(self.0.as_bytes());
        serializer.serialize_str(&hex_string)
    }
}
impl<'de> Deserialize<'de> for Address {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let hex_string = String::deserialize(deserializer)?;
        let bytes = hex::decode(&hex_string).map_err(serde::de::Error::custom)?;

        if bytes.len() != 32 {
            return Err(serde::de::Error::custom(format!(
                "Address must be 32 bytes, got {}",
                bytes.len()
            )));
        }

        let mut array = [0u8; 32];
        array.copy_from_slice(&bytes);

        VerifyingKey::from_bytes(&array)
            .map(Self)
            .map_err(|e| serde::de::Error::custom(format!("Invalid address: {e}")))
    }
}
// Custom thiserror for Address-related errors
#[derive(Debug, thiserror::Error)]
pub enum AddressError {
    #[error("Invalid address bytes")]
    InvalidAddress,
    #[error("Signature error: {0}")]
    SignatureError(#[from] SignatureError),
}
