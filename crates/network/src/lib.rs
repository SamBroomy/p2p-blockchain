mod behaviour;
mod config;
mod messages;

pub use behaviour::{
    BlockchainBehaviour, BlockchainBehaviourEvent, SyncRequest, SyncResponse, TOPIC_BLOCKS,
    TOPIC_TRANSACTIONS,
};
pub use config::NetworkConfig;
// Re-export libp2p types that consumers will need
pub use libp2p::{
    Multiaddr, PeerId, gossipsub, mdns,
    swarm::{Swarm, SwarmEvent},
};
pub use messages::{NetworkError, NetworkMessage};
