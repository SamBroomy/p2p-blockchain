use std::time::Duration;

use blake3::Hash;
use blockchain_types::Block;
use libp2p::{
    PeerId, StreamProtocol, gossipsub,
    gossipsub::{MessageId, ValidationMode},
    mdns, request_response,
    request_response::ProtocolSupport,
    swarm::NetworkBehaviour,
};
use serde::{Deserialize, Serialize};

use crate::messages::NetworkError;

/// Topics for different message types
pub const TOPIC_TRANSACTIONS: &str = "blockchain/transactions";
pub const TOPIC_BLOCKS: &str = "blockchain/blocks";
pub const SYNC_STREAM: &str = "/blockchain-sync/1.0.0";

/// Custom network behaviour combining Gossipsub and mDNS
#[derive(NetworkBehaviour)]
pub struct BlockchainBehaviour {
    pub gossipsub: gossipsub::Behaviour,
    pub mdns: mdns::tokio::Behaviour,
    pub request_response: request_response::json::Behaviour<SyncRequest, SyncResponse>,
}

/// Request types for direct peer-to-peer communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncRequest {
    /// Request specific blocks by hash
    Blocks(Vec<Hash>),
    /// Request status (height, tip)
    Status,
    /// Request blocks in a range
    BlockRange { from: u64, to: u64 },
}

/// Response types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncResponse {
    Blocks(Vec<Block>),
    Status {
        height: u64,
        tip_hash: Hash,
        cumulative_difficulty: u128,
    },
}

impl BlockchainBehaviour {
    pub fn new(key: &libp2p::identity::Keypair) -> Result<Self, NetworkError> {
        // Content-address messages by their hash (prevents duplicate propagation)
        let message_id_fn = |message: &gossipsub::Message| {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::hash::DefaultHasher::new();
            message.data.hash(&mut hasher);
            MessageId::from(hasher.finish().to_string())
        };

        // Configure Gossipsub
        let gossipsub_config = gossipsub::ConfigBuilder::default()
            .heartbeat_interval(Duration::from_secs(10))
            .validation_mode(ValidationMode::Strict) // Enforce message signing
            .message_id_fn(message_id_fn)
            .build()
            .map_err(|e| NetworkError::Libp2p(format!("Gossipsub config error: {e}")))?;

        // Build gossipsub behaviour
        let gossipsub = gossipsub::Behaviour::new(
            gossipsub::MessageAuthenticity::Signed(key.clone()),
            gossipsub_config,
        )
        .map_err(|e| NetworkError::Libp2p(format!("Gossipsub creation error: {e}")))?;

        // Build mDNS behaviour for local peer discovery
        let mdns = mdns::tokio::Behaviour::new(mdns::Config::default(), key.public().to_peer_id())
            .map_err(|e| NetworkError::Libp2p(format!("mDNS creation error: {e}")))?;

        // Configure request-response protocol for block syncing
        let request_response = request_response::json::Behaviour::new(
            [(StreamProtocol::new(SYNC_STREAM), ProtocolSupport::Full)],
            request_response::Config::default(),
        );

        Ok(Self {
            gossipsub,
            mdns,
            request_response,
        })
    }

    /// Subscribe to both transaction and block topics
    pub fn subscribe_to_topics(&mut self) -> Result<(), NetworkError> {
        let tx_topic = gossipsub::IdentTopic::new(TOPIC_TRANSACTIONS);
        let block_topic = gossipsub::IdentTopic::new(TOPIC_BLOCKS);

        self.gossipsub
            .subscribe(&tx_topic)
            .map_err(|e| NetworkError::Libp2p(format!("Subscribe error: {e}")))?;

        self.gossipsub
            .subscribe(&block_topic)
            .map_err(|e| NetworkError::Libp2p(format!("Subscribe error: {e}")))?;
        Ok(())
    }

    /// Publish a message to a specific topic
    pub fn publish(&mut self, topic: &str, data: Vec<u8>) -> Result<MessageId, NetworkError> {
        let topic = gossipsub::IdentTopic::new(topic);
        self.gossipsub
            .publish(topic, data)
            .map_err(|e| NetworkError::Libp2p(format!("Publish error: {e}")))
    }

    /// Add a peer discovered via mDNS to Gossipsub
    pub fn add_peer(&mut self, peer_id: &PeerId) {
        self.gossipsub.add_explicit_peer(peer_id);
    }

    /// Remove an expired peer from Gossipsub
    pub fn remove_peer(&mut self, peer_id: &PeerId) {
        self.gossipsub.remove_explicit_peer(peer_id);
    }

    pub fn list_peers(&self) -> impl Iterator<Item = &PeerId> {
        self.gossipsub.peer_protocol().map(|(peer_id, _)| peer_id)
    }
}
