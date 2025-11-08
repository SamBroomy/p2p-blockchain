use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    time::Duration,
};

use libp2p::{
    PeerId,
    gossipsub::{self, MessageId, ValidationMode},
    mdns,
    swarm::NetworkBehaviour,
};

use crate::messages::NetworkError;

/// Topics for different message types
pub const TOPIC_TRANSACTIONS: &str = "blockchain/transactions";
pub const TOPIC_BLOCKS: &str = "blockchain/blocks";
pub const TOPIC_SYNC: &str = "blockchain/sync";

/// Custom network behaviour combining Gossipsub and mDNS
#[derive(NetworkBehaviour)]
pub struct BlockchainBehaviour {
    pub gossipsub: gossipsub::Behaviour,
    pub mdns: mdns::tokio::Behaviour,
}

impl BlockchainBehaviour {
    pub fn new(key: &libp2p::identity::Keypair) -> Result<Self, NetworkError> {
        // Content-address messages by their hash (prevents duplicate propagation)
        let message_id_fn = |message: &gossipsub::Message| {
            let mut hasher = DefaultHasher::new();
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

        Ok(Self { gossipsub, mdns })
    }

    /// Subscribe to both transaction and block topics
    pub fn subscribe_to_topics(&mut self) -> Result<(), NetworkError> {
        let tx_topic = gossipsub::IdentTopic::new(TOPIC_TRANSACTIONS);
        let block_topic = gossipsub::IdentTopic::new(TOPIC_BLOCKS);
        let sync_topic = gossipsub::IdentTopic::new(TOPIC_SYNC);

        self.gossipsub
            .subscribe(&tx_topic)
            .map_err(|e| NetworkError::Libp2p(format!("Subscribe error: {e}")))?;

        self.gossipsub
            .subscribe(&block_topic)
            .map_err(|e| NetworkError::Libp2p(format!("Subscribe error: {e}")))?;
        self.gossipsub
            .subscribe(&sync_topic)
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
}
