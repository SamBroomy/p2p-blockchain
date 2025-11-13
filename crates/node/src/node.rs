use core::fmt;
use std::{collections::HashMap, error::Error, time::Duration};

use blake3::Hash;
use blockchain::{BlockAddResult, BlockChain};
use blockchain_types::{
    Block, BlockConstructor, ConstMiner, Miner, Transaction,
    wallet::{Address, Wallet},
};
use libp2p::{
    PeerId,
    futures::StreamExt,
    gossipsub, mdns,
    request_response::{self, InboundRequestId, OutboundRequestId},
};
use mempool::Mempool;
use network::{
    BlockchainBehaviour, BlockchainBehaviourEvent, NetworkMessage, Swarm, SwarmEvent, SyncRequest,
    SyncResponse, TOPIC_BLOCKS, TOPIC_TRANSACTIONS,
};
use tracing::{debug, info, warn};

pub struct Node {
    pub blockchain: BlockChain<Miner>,
    pub mempool: Mempool,
    wallet: Wallet,
    pub swarm: Swarm<BlockchainBehaviour>,
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
            // .with_quic()
            .with_behaviour(|key| Ok(BlockchainBehaviour::new(key)?))?
            .build();

        // Subscribe to topics
        swarm.behaviour_mut().subscribe_to_topics()?;

        // Listen on all interfaces
        // swarm.listen_on("/ip4/0.0.0.0/udp/0/quic-v1".parse()?)?;
        swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;

        let blockchain: BlockChain<Miner> = BlockChain::new(5);

        Ok(Self {
            blockchain,
            mempool: Mempool::new(),
            wallet,
            swarm,

            pending_requests: HashMap::new(),
        })
    }

    /// Main event loop
    pub async fn run(&mut self) {
        // TODO: on startup, get the blockchain state from peers (first sync)

        // Timer for periodic sync checks
        let mut sync_interval = tokio::time::interval(Duration::from_secs(10));

        debug!("Node event loop started");

        loop {
            tokio::select! {
                _ = sync_interval.tick() => {
                    debug!("Performing periodic sync check");
                    self.sync_check();
                }
                event = self.swarm.select_next_some() => {
                    debug!("Swarm event received");
                    self.handle_swarm_event(event);

                }
            }
        }
    }

    pub fn handle_swarm_event(&mut self, event: SwarmEvent<BlockchainBehaviourEvent>) {
        match event {
            // New listening address
            SwarmEvent::NewListenAddr { address, .. } => {
                info!(address = %address, "Listening on new address");
            }

            // mDNS discovered a peer
            SwarmEvent::Behaviour(event) => {
                use network::BlockchainBehaviourEvent;

                match event {
                    BlockchainBehaviourEvent::Mdns(mdns::Event::Discovered(peers)) => {
                        for (peer_id, _) in peers {
                            debug!(peer_id = %peer_id, "mDNS discovered a new peer");
                            self.swarm.behaviour_mut().add_peer(&peer_id);
                            self.request_peer_status(&peer_id);
                        }
                    }

                    BlockchainBehaviourEvent::Mdns(mdns::Event::Expired(peers)) => {
                        for (peer_id, _) in peers {
                            debug!(peer_id = %peer_id, "mDNS peer has expired");
                            self.swarm.behaviour_mut().remove_peer(&peer_id);
                        }
                    }

                    BlockchainBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                        propagation_source: peer_id,
                        message_id: id,
                        message,
                    }) => {
                        debug!(peer_id = %peer_id, id = %id, "Gossipsub message received");
                        debug!(message = %String::from_utf8_lossy(&message.data), "Gossipsub message data");
                        // Deserialize and handle message
                        if let Ok(msg) = NetworkMessage::from_bytes(&message.data) {
                            self.handle_network_message(msg);
                        }
                    }
                    BlockchainBehaviourEvent::RequestResponse(
                        request_response::Event::Message {
                            peer,
                            connection_id,
                            message,
                        },
                    ) => {
                        debug!(peer = %peer, connection_id = %connection_id, "RequestResponse message received");
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
                info!("Peer requested our status");
                let status = SyncResponse::Status {
                    height: self.blockchain.main_chain_len() as u64,
                    tip_hash: *self.blockchain.latest_block_hash(),
                    cumulative_difficulty: self.blockchain.cumulative_difficulty(),
                };
                info!(
                    status = ?status,
                    "Responding with status"
                );
                status
            }
            SyncRequest::Blocks(hashes) => {
                info!(
                    blocks_requested = hashes.len(),
                    "Peer requested specific blocks"
                );

                let mut blocks = Vec::new();
                for hash in hashes {
                    if let Some(block) = self.blockchain.get_block(&hash).cloned() {
                        blocks.push(block);
                    }
                }
                SyncResponse::Blocks(blocks)
            }
            SyncRequest::BlockRange { from, to } => {
                info!(from = from, to = to, "Peer requested block range");
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
                    info!(
                        height = height,
                        tip_hash = ?tip_hash,
                        cumulative_difficulty = cumulative_difficulty,
                        "Received status response"
                    );
                    let our_height = self.blockchain.main_chain_len() as u64;
                    let our_difficulty = self.blockchain.cumulative_difficulty();
                    // TODO
                    if height > our_height || cumulative_difficulty > our_difficulty {
                        info!(
                            peer_height = height,
                            peer_difficulty = cumulative_difficulty,
                            our_height = our_height,
                            our_difficulty = our_difficulty,
                            "Peer is ahead, requesting blocks"
                        );
                        // Request blocks from our height to their height
                        self.request_block_range(our_height, height);
                    }
                }
                (SyncRequestType::Blocks(_hashes), SyncResponse::Blocks(blocks)) => {
                    // TODO: Check we got the blocks we requested

                    info!(
                        blocks_received = blocks.len(),
                        "Received blocks in response"
                    );
                    for block in blocks {
                        match self.blockchain.add_block(block) {
                            BlockAddResult::Added { processed_orphans } => {
                                info!(
                                    processed_orphans = processed_orphans,
                                    "Added block, processed orphans"
                                );
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
                    warn!("Mismatched request and response types");
                }
            }
        } else {
            warn!("Request ID not found");
        }
    }

    fn handle_network_message(&mut self, msg: NetworkMessage) {
        match msg {
            NetworkMessage::Transaction(tx) => {
                info!(tx_hash = ?tx.hash(), "Received transaction from network");
                // Validate and add to mempool
                if self.blockchain.validate_transaction(&tx) {
                    self.mempool.add_transaction(tx);
                }
            }

            NetworkMessage::Block(block) => {
                info!(block_hash = ?block.hash(), "Received block from network");
                if let BlockAddResult::Added { .. } = self.blockchain.add_block(block.clone()) {
                    // Remove included transactions from mempool
                    self.mempool.remove_transactions(block.transactions());
                }
            }
        }
    }

    /// Periodically check if we're in sync
    pub fn sync_check(&mut self) {
        debug!("Running sync check");
        // Check 1: Too many orphans?
        if self.blockchain.orphan_count() > 2 {
            debug!(
                orphan_count = self.blockchain.orphan_count(),
                "Orphan count high, requesting missing blocks"
            );
            let missing = self.blockchain.get_missing_blocks();
            self.request_blocks(missing);
        }

        // Check 2: Ask a random peer for their status
        self.request_random_peer_status();
    }

    pub fn list_peers(&self) -> impl Iterator<Item = &PeerId> + '_ {
        self.swarm.behaviour().list_peers()
    }

    pub fn get_random_peer(&self) -> Option<PeerId> {
        use rand::seq::IteratorRandom;
        self.list_peers().choose(&mut rand::thread_rng()).copied()
    }

    fn request_peer_status(&mut self, peer: &PeerId) {
        info!(peer = %peer, "Requesting status from peer");
        let request_id = self
            .swarm
            .behaviour_mut()
            .request_response
            .send_request(peer, SyncRequest::Status);
        self.pending_requests
            .insert(request_id, SyncRequestType::Status);
    }

    /// Request status from ONE random peer
    fn request_random_peer_status(&mut self) {
        let random_peer = self.get_random_peer();
        let Some(peer) = random_peer else {
            debug!("No peers to request status from");
            return;
        };
        self.request_peer_status(&peer);
    }

    /// Request specific blocks from ONE random peer
    fn request_blocks(&mut self, hashes: Vec<Hash>) {
        assert!(!hashes.is_empty());
        let random_peer = self.get_random_peer();
        let Some(peer) = random_peer else {
            warn!("No peers to request blocks from");
            return;
        };

        info!(peer = %peer, blocks_requested = hashes.len(), "Requesting blocks from peer");
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
            warn!("No peers to request block range from");
            return;
        };
        info!(from_height = from_height, to_height = to_height, peer = %peer, "Requesting block range from peer");

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
    pub fn broadcast_transaction(&mut self, tx: Transaction) -> Result<(), NodeError> {
        let msg = NetworkMessage::Transaction(tx);
        let bytes = msg.to_bytes()?;

        self.swarm
            .behaviour_mut()
            .publish(TOPIC_TRANSACTIONS, bytes)?;
        Ok(())
    }

    /// Broadcast a block to the network
    pub fn broadcast_block(&mut self, block: Block) -> Result<(), NodeError> {
        let msg = NetworkMessage::Block(block);
        let bytes = msg.to_bytes()?;

        self.swarm.behaviour_mut().publish(TOPIC_BLOCKS, bytes)?;

        Ok(())
    }

    //  ---- Broadcasting Methods ----
    /// Create and broadcast a transaction to the network
    pub fn send_transaction(
        &mut self,
        receiver_address: &Address,
        amount: u64,
    ) -> Result<Transaction, NodeError> {
        // 1. Check available balance accounting for pending transactions
        let balance_info = self.get_balance_info();
        if amount > balance_info.available {
            warn!(
                amount = amount,
                available = balance_info.available,
                "Insufficient balance to send transaction"
            );
            return Err(NodeError::InsufficientBalance {
                requested: amount,
                available: balance_info.available,
                pending: balance_info.pending_sends,
            });
        }
        // 2. Create the transaction
        let tx = self.wallet.create_transaction(receiver_address, amount);
        // 3. Double-check against blockchain state (should always pass after step 1)
        // This catches race conditions or state inconsistencies
        debug_assert!(
            self.blockchain.validate_transaction(&tx),
            "Transaction should be valid after balance check"
        );
        // 4. Add to the local mempool
        self.mempool.add_transaction(tx.clone());
        // 5. Broadcast to network (best-effort, ignore no-peers errors)

        if let Err(e) = self.broadcast_transaction(tx.clone()) {
            debug!(error = %e, "Failed to broadcast block (no peers?)");
        }
        info!(
            tx_hash = ?tx.hash(),
            receiver_address = %receiver_address,
            amount = amount,
            "Transaction created and broadcast"
        );
        Ok(tx)
    }

    pub fn mine_block(&mut self) -> Result<Block, NodeError> {
        // When we mine should we look at the block just mined, find all txs in the mempool relating to that tx and remove it? Like we need to clean the mempool occasionally from successful txs
        // 1. Get transactions from mempool
        let mut txs = Vec::new();
        while let Some(tx) = self.mempool.get_transaction() {
            // Locally validated (need to validate in order as well but the miner should do that)
            if self.blockchain.validate_transaction(&tx) {
                txs.push(tx);
                if txs.len() >= 10 {
                    break; // limit to 10 transactions per block for now
                }
            }
        }

        // 2. Create new block
        info!(tx_count = txs.len(), "Mining new block with transactions");
        let previous_hash = *self.blockchain.latest_block_hash();
        let index = self.blockchain.height();
        let constructor =
            BlockConstructor::new(index, &txs, previous_hash, Some(*self.wallet.address()));

        let mined_block = self.blockchain.mine(constructor);

        info!(block_hash = ?mined_block.hash(), "Mined new block");

        // 4. Add to our own blockchain
        match self.blockchain.add_block(mined_block.clone()) {
            BlockAddResult::Added { .. } => {
                // 5. Broadcast to network
                if let Err(e) = self.broadcast_block(mined_block.clone()) {
                    debug!(error = %e, "Failed to broadcast block");
                }
                Ok(mined_block)
            }
            BlockAddResult::Orphaned { missing_parent } => {
                // This should not happen since we just mined it
                info!(missing_parent = ?missing_parent, "Mined block is orphaned, missing parent");
                Err(NodeError::BlockRejected)
            }
            BlockAddResult::Rejected(reason) => {
                warn!(reason = ?reason, "Mined block was rejected");
                Err(NodeError::BlockRejected)
            }
        }
    }

    pub fn my_address(&self) -> Address {
        *self.wallet.address()
    }

    pub fn get_balance_info(&self) -> BalanceInfo {
        let confirmed = self.blockchain.get_balance(self.wallet.address());

        let pending_sends = self
            .mempool
            .pending_transactions()
            .filter_map(|tx| {
                if let Some(addr) = tx.sender()
                    && addr == self.wallet.address()
                {
                    Some(tx.amount())
                } else {
                    None
                }
            })
            .sum();
        let available = confirmed.saturating_sub(pending_sends);

        BalanceInfo {
            confirmed,
            pending_sends,
            available,
        }
    }
}

#[derive(Debug)]
pub struct BalanceInfo {
    pub confirmed: u64,
    pub pending_sends: u64,
    pub available: u64,
}

impl fmt::Display for BalanceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Confirmed: {}, Pending Sends: {}, Available: {}",
            self.confirmed, self.pending_sends, self.available
        )
    }
}

#[derive(Debug, thiserror::Error)]
pub enum NodeError {
    #[error("Insufficient balance")]
    InsufficientBalance {
        requested: u64,
        available: u64,
        pending: u64,
    },
    #[error("Transaction validation failed")]
    TransactionValidationFailed,
    #[error("Block rejected")]
    BlockRejected,
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    #[error("Network error: {0}")]
    NetworkError(#[from] network::NetworkError),
}
