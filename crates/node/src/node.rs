use std::{collections::HashMap, error::Error, time::Duration};

use blake3::Hash;
use blockchain::{BlockAddResult, BlockChain};
use blockchain_types::{Block, Transaction, wallet::Wallet};
use libp2p::{
    PeerId,
    futures::StreamExt,
    gossipsub, mdns,
    request_response::{self, InboundRequestId, OutboundRequestId},
};
use mempool::Mempool;
use network::{
    BlockchainBehaviour, NetworkMessage, Swarm, SwarmEvent, SyncRequest, SyncResponse,
    TOPIC_BLOCKS, TOPIC_TRANSACTIONS,
};
use tokio::{io, io::AsyncBufReadExt};

pub struct Node {
    blockchain: BlockChain,
    mempool: Mempool,
    wallet: Wallet,
    swarm: Swarm<BlockchainBehaviour>,
    // Trackpending requests (for responses)
    pending_requests: HashMap<OutboundRequestId, SyncRequestType>,
}

#[derive(Debug)]
enum SyncRequestType {
    Status,
    Blocks(Vec<Hash>),
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

            pending_requests: HashMap::new(),
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
                                        self.swarm.behaviour_mut()
                                        .add_peer(&peer_id);
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
                                BlockchainBehaviourEvent::RequestResponse(
                                    request_response::Event::Message { peer, connection_id, message  },
                                ) => {
                                    println!("Received request response message from peer {peer} with connection id {connection_id}");
                                    // Handle incoming request
                                    self.handle_request_response_message(message);
                                }
                                _ => {
                                    // debug!("Unhandled behaviour event: {:?}", event);
                                }
                            }
                        }

                        _ => {}
                    }
                }
            }
        }
    }

    fn handle_request_response_message(
        &mut self,
        msg: request_response::Message<SyncRequest, SyncResponse>,
    ) {
        match msg {
            request_response::Message::Request {
                request_id,
                request,
                channel,
            } => {
                // Handle incoming request
                self.handle_sync_request(request_id, request, channel);
            }
            request_response::Message::Response {
                request_id,
                response,
            } => {
                // Handle incoming response
                self.handle_sync_response(request_id, response);
            }
        }
    }

    fn handle_sync_request(
        &mut self,
        _request_id: InboundRequestId,
        request: SyncRequest,
        channel: request_response::ResponseChannel<SyncResponse>,
    ) {
        let response = match request {
            SyncRequest::Status => {
                println!("Peer requested our status");
                SyncResponse::Status {
                    height: self.blockchain.main_chain_len() as u64,
                    tip_hash: *self.blockchain.latest_block_hash(),
                    cumulative_difficulty: self.blockchain.cumulative_difficulty(),
                }
            }
            SyncRequest::Blocks(hashes) => {
                println!("Peer requested {} blocks", hashes.len());
                let mut blocks = Vec::new();
                for hash in hashes {
                    if let Some(block) = self.blockchain.get_block(&hash).cloned() {
                        blocks.push(block);
                    }
                }
                SyncResponse::Blocks(blocks)
            }
            SyncRequest::BlockRange { from, to } => {
                println!("Peer requested blocks from height {from} to {to}");
                let blocks = self.blockchain.get_blocks_in_range(from, to);
                SyncResponse::Blocks(blocks)
            }
        };
        self.swarm
            .behaviour_mut()
            .request_response
            .send_response(channel, response)
            .unwrap();
    }

    fn handle_sync_response(&mut self, request_id: OutboundRequestId, response: SyncResponse) {
        if let Some(request_type) = self.pending_requests.remove(&request_id) {
            match (request_type, response) {
                (
                    SyncRequestType::Status,
                    SyncResponse::Status {
                        height,
                        tip_hash,
                        cumulative_difficulty,
                    },
                ) => {
                    println!(
                        "Received status response: height={height}, tip_hash={tip_hash:?}, cumulative_difficulty={cumulative_difficulty}"
                    );
                    let our_height = self.blockchain.main_chain_len() as u64;
                    let our_difficulty = self.blockchain.cumulative_difficulty();

                    if height > our_height || cumulative_difficulty > our_difficulty {
                        println!(
                            "Peer is ahead! Them: height={height}, diff={cumulative_difficulty}"
                        );
                        println!(
                            "                  Us:   height={our_height}, diff={our_difficulty}"
                        );

                        // Request blocks from our height to their height
                        self.request_block_range(our_height, height);
                    }
                }
                (SyncRequestType::Blocks(_hashes), SyncResponse::Blocks(blocks)) => {
                    // TODO: Check we got the blocks we requested

                    println!(
                        "Received {} blocks in response to our request",
                        blocks.len()
                    );
                    for block in blocks {
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

                _ => {
                    println!("Mismatched request and response types");
                }
            }
        } else {
            println!("Received response for unknown request ID");
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

    fn get_random_peer(&self) -> Option<PeerId> {
        use rand::seq::IteratorRandom;
        self.swarm
            .behaviour()
            .list_peers()
            .choose(&mut rand::thread_rng())
            .copied()
    }

    /// Request status from ONE random peer
    fn request_peer_status(&mut self) {
        let random_peer = self.get_random_peer();
        let Some(peer) = random_peer else {
            println!("No peers to request status from");
            return;
        };

        println!("Requesting status from peer {peer}");
        let request_id = self
            .swarm
            .behaviour_mut()
            .request_response
            .send_request(&peer, SyncRequest::Status);
        self.pending_requests
            .insert(request_id, SyncRequestType::Status);
    }

    /// Request specific blocks from ONE random peer
    fn request_blocks(&mut self, hashes: Vec<Hash>) {
        assert!(!hashes.is_empty());
        let random_peer = self.get_random_peer();
        let Some(peer) = random_peer else {
            println!("No peers to request blocks from");
            return;
        };

        println!("Requesting {} blocks from peer {peer}", hashes.len());

        let request_id = self
            .swarm
            .behaviour_mut()
            .request_response
            .send_request(&peer, SyncRequest::Blocks(hashes.clone()));

        self.pending_requests
            .insert(request_id, SyncRequestType::Blocks(hashes));
    }

    fn request_block_range(&mut self, from_height: u64, to_height: u64) {
        debug_assert!(to_height > from_height);
        let random_peer = self.get_random_peer();
        let Some(peer) = random_peer else {
            println!("No peers to request block range from");
            return;
        };

        println!("Requesting blocks from height {from_height} to {to_height} from peer {peer}");

        let request_id = self.swarm.behaviour_mut().request_response.send_request(
            &peer,
            SyncRequest::BlockRange {
                from: from_height,
                to: to_height,
            },
        );

        // We don't track the request type here since the response will be blocks
        self.pending_requests
            .insert(request_id, SyncRequestType::Blocks(Vec::new()));
    }

    /// Helper: Get blocks by hash
    fn get_blocks_by_hash(&self, hashes: Vec<Hash>) -> Vec<Block> {
        let mut blocks = Vec::new();
        for hash in hashes {
            if let Some(block) = self.blockchain.get_block(&hash).cloned() {
                blocks.push(block);
            }
        }
        blocks
    }

    /// Helper: Get blocks in a height range
    fn get_blocks_by_range(&self, from: u64, to: u64) -> Vec<Block> {
        self.blockchain.get_blocks_in_range(from, to)
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
