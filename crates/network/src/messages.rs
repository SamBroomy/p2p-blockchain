use blockchain_types::{Block, Transaction};
use serde::{Deserialize, Serialize};

/// Messages broadcast over the P2P network
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)]
pub enum NetworkMessage {
    /// New transaction to be added to mempool
    Transaction(Transaction),
    /// New mined block
    Block(Block),
}
impl NetworkMessage {
    /// Serialize to JSON bytes for gossipsub
    pub fn to_bytes(&self) -> Result<Vec<u8>, NetworkError> {
        serde_json::to_vec(self).map_err(NetworkError::Serialization)
    }

    /// Deserialize from JSON bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, NetworkError> {
        serde_json::from_slice(bytes).map_err(NetworkError::Deserialization)
    }
}
#[derive(Debug, thiserror::Error)]
pub enum NetworkError {
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Deserialization error: {0}")]
    Deserialization(serde_json::Error),

    #[error("libp2p error: {0}")]
    Libp2p(String),
}
