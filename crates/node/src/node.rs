use std::{error::Error, time::Duration};

use blake3::Hash;
use blockchain::{BlockAddResult, BlockChain};
use blockchain_types::{Block, Transaction, wallet::Wallet};
use libp2p::{futures::StreamExt, gossipsub, mdns};
use mempool::Mempool;
use network::{
    BlockchainBehaviour, NetworkMessage, Swarm, SwarmEvent, TOPIC_BLOCKS, TOPIC_SYNC,
    TOPIC_TRANSACTIONS,
};
use tokio::{io, io::AsyncBufReadExt};

pub struct Node {
    blockchain: BlockChain,
    mempool: Mempool,
    wallet: Wallet,
    swarm: Swarm<BlockchainBehaviour>,
}

impl Node {
    pub fn new(wallet: Wallet) -> Result<Self, Box<dyn Error>> {
        // Build the swarm
        let mut swarm = libp2p::SwarmBuilder::with_new_identity()
            .with_tokio()
            .with_tcp(
                libp2p::tcp::Config::default(),
                libp2p::noise::Config::new,
                libp2p::yamux::Config::default,
            )?
            .with_quic()
            .with_behaviour(|key| Ok(BlockchainBehaviour::new(key)?))?
            .build();

        // Subscribe to topics
        swarm.behaviour_mut().subscribe_to_topics()?;

        // Listen on all interfaces
        swarm.listen_on("/ip4/0.0.0.0/udp/0/quic-v1".parse()?)?;
        swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;

        Ok(Self {
            blockchain: BlockChain::new(4), // difficulty 4
            mempool: Mempool::new(),
            wallet,
            swarm,
        })
    }

    /// Main event loop
    pub async fn run(&mut self) {
        // Read full lines from stdin
        let mut _stdin = io::BufReader::new(io::stdin()).lines();

        // Timer for periodic sync checks
        let mut sync_interval = tokio::time::interval(Duration::from_secs(30));

        loop {
            tokio::select! {
                _ = sync_interval.tick() => {
                    self.sync_check();
                }
                event = self.swarm.select_next_some() => {
                    match event {
                        // New listening address
                        SwarmEvent::NewListenAddr { address, .. } => {
                            println!("📡 Listening on {address}");
                        }

                        // mDNS discovered a peer
                        SwarmEvent::Behaviour(event) => {
                            use network::BlockchainBehaviourEvent;

                            match event {
                                BlockchainBehaviourEvent::Mdns(mdns::Event::Discovered(peers)) => {
                                    for (peer_id, _) in peers {
                                        println!("mDNS discovered a new peer: {peer_id}");
                                        self.swarm.behaviour_mut().add_peer(&peer_id);
                                    }
                                }

                                BlockchainBehaviourEvent::Mdns(mdns::Event::Expired(peers)) => {
                                    for (peer_id, _) in peers {
                                        println!("mDNS discover peer has expired: {peer_id}");
                                        self.swarm.behaviour_mut().remove_peer(&peer_id);
                                    }
                                }

                                BlockchainBehaviourEvent::Gossipsub(
                                    gossipsub::Event::Message {
                                        propagation_source: peer_id,
                                        message_id: id,
                                        message,
                                    },
                                ) => {
                                    println!(
                                        "Got message: '{}' with id: {id} from peer: {peer_id}",
                                        String::from_utf8_lossy(&message.data),
                                    );
                                    // Deserialize and handle message
                                    if let Ok(msg) = NetworkMessage::from_bytes(&message.data) {
                                        self.handle_network_message(msg);
                                    }
                                }

                                _ => {}
                            }
                        }

                        _ => {}
                    }
                }
            }
        }
    }

    fn handle_network_message(&mut self, msg: NetworkMessage) {
        match msg {
            NetworkMessage::Transaction(tx) => {
                println!("Received transaction: {:?}", tx.hash());
                // Validate and add to mempool
                if self.blockchain.validate_transaction(&tx) {
                    self.mempool.add_transaction(tx);
                }
            }

            NetworkMessage::Block(block) => {
                println!("Received block: {:?}", block.hash());
                if let BlockAddResult::Added { .. } = self.blockchain.add_block(block.clone()) {
                    // Remove included transactions from mempool
                    self.mempool.remove_transactions(block.transactions());
                }
            }

            NetworkMessage::ResponseBlockRange(blocks) => {
                println!("📥 Received {} blocks", blocks.len());
                // Add blocks to blockchain
                for block in blocks {
                    let _ = self.blockchain.add_block(block);
                }
            }

            NetworkMessage::RequestBlocks(hashes) => {
                println!("Peer requested {} blocks", hashes.len());
                // Look up blocks in our blockchain and respond
                let blocks = self.get_blocks_by_hash(hashes);
                if !blocks.is_empty() {
                    let response = NetworkMessage::ResponseBlocks(blocks);
                    if let Ok(bytes) = response.to_bytes() {
                        let _ = self.swarm.behaviour_mut().publish(TOPIC_SYNC, bytes);
                    }
                }
            }

            NetworkMessage::ResponseBlocks(blocks) => {
                println!("Received {} blocks from peer", blocks.len());
                for block in blocks {
                    // Try to add each block
                    match self.blockchain.add_block(block) {
                        BlockAddResult::Added { processed_orphans } => {
                            println!("✅ Added block, processed {processed_orphans} orphans");
                        }
                        BlockAddResult::Orphaned { missing_parent } => {
                            // Still missing parent, request it
                            self.request_blocks(vec![missing_parent]);
                        }
                        BlockAddResult::Rejected(_) => {}
                    }
                }
            }

            NetworkMessage::RequestStatus => {
                // Someone wants to know our status
                let response = NetworkMessage::ResponseStatus {
                    chain_height: self.blockchain.main_chain_len() as u64,
                    tip_hash: *self.blockchain.latest_block_hash(),
                    cumulative_difficulty: self.blockchain.cumulative_difficulty(),
                };
                if let Ok(bytes) = response.to_bytes() {
                    let _ = self.swarm.behaviour_mut().publish(TOPIC_SYNC, bytes);
                }
            }

            NetworkMessage::ResponseStatus {
                chain_height,
                tip_hash,
                cumulative_difficulty,
            } => {
                let our_height = self.blockchain.main_chain_len() as u64;
                let our_difficulty = self.blockchain.cumulative_difficulty();

                if chain_height > our_height || cumulative_difficulty > our_difficulty {
                    println!(
                        "Peer is ahead! Them: height={chain_height}, diff={cumulative_difficulty}"
                    );
                    println!("                  Us:   height={our_height}, diff={our_difficulty}");

                    // Request blocks from our height to their height
                    self.request_block_range(our_height, chain_height);
                }
            }

            NetworkMessage::RequestBlockRange {
                from_height,
                to_height,
            } => {
                println!("Peer requested blocks from {from_height} to {to_height}");
                let blocks = self.get_blocks_by_range(from_height, to_height);
                if !blocks.is_empty() {
                    let response = NetworkMessage::ResponseBlockRange(blocks);
                    if let Ok(bytes) = response.to_bytes() {
                        let _ = self.swarm.behaviour_mut().publish(TOPIC_SYNC, bytes);
                    }
                }
            }
        }
    }

    /// Periodically check if we're in sync
    fn sync_check(&mut self) {
        // Check 1: Too many orphans?
        if self.blockchain.orphan_count() > 3 {
            println!(
                "Orphan pool has {} blocks, requesting parents...",
                self.blockchain.orphan_count()
            );
            let missing = self.blockchain.get_missing_blocks();
            self.request_blocks(missing);
        }

        // Check 2: Ask a random peer for their status
        self.request_peer_status();
    }

    /// Request missing blocks from network
    fn request_blocks(&mut self, hashes: Vec<Hash>) {
        if hashes.is_empty() {
            return;
        }

        println!("Requesting {} missing blocks from peers", hashes.len());
        let msg = NetworkMessage::RequestBlocks(hashes);
        if let Ok(bytes) = msg.to_bytes() {
            // Publish on a "sync" topic
            let _ = self.swarm.behaviour_mut().publish(TOPIC_SYNC, bytes);
        }
    }

    /// Request status from all peers
    fn request_peer_status(&mut self) {
        let msg = NetworkMessage::RequestStatus;
        if let Ok(bytes) = msg.to_bytes() {
            let _ = self.swarm.behaviour_mut().publish(TOPIC_SYNC, bytes);
        }
    }

    /// Helper: Get blocks by hash
    fn get_blocks_by_hash(&self, hashes: Vec<Hash>) -> Vec<Block> {
        // TODO: Blockchain needs a method to look up blocks by hash
        // For now, return empty
        Vec::new()
    }

    /// Helper: Get blocks in a height range
    fn get_blocks_by_range(&self, from: u64, to: u64) -> Vec<Block> {
        // TODO: Blockchain needs a method to get blocks by index
        // For now, return empty
        Vec::new()
    }

    fn request_block_range(&mut self, from_height: u64, to_height: u64) {
        let msg = NetworkMessage::RequestBlockRange {
            from_height,
            to_height,
        };
        if let Ok(bytes) = msg.to_bytes() {
            let _ = self.swarm.behaviour_mut().publish(TOPIC_SYNC, bytes);
        }
    }

    /// Broadcast a transaction to the network
    pub fn broadcast_transaction(&mut self, tx: Transaction) {
        let msg = NetworkMessage::Transaction(tx);
        if let Ok(bytes) = msg.to_bytes() {
            if let Ok(msg_is) = self
                .swarm
                .behaviour_mut()
                .publish(TOPIC_TRANSACTIONS, bytes)
            {
                println!("Broadcasted transaction with message ID: {msg_is}");
            } else {
                eprintln!("Failed to broadcast transaction");
            }
        }
    }

    /// Broadcast a block to the network
    pub fn broadcast_block(&mut self, block: Block) {
        let msg = NetworkMessage::Block(block);
        if let Ok(bytes) = msg.to_bytes() {
            if let Ok(msg_id) = self.swarm.behaviour_mut().publish(TOPIC_BLOCKS, bytes) {
                println!("Broadcasted block with message ID: {msg_id}");
            } else {
                eprintln!("Failed to broadcast block");
            }
        }
    }
}
