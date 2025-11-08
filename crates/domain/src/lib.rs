#![feature(assert_matches)]

mod blockchain;
mod crypto;
mod mempool;

pub use blockchain::{Block, BlockChain, BlockConstructor, Transaction, Wallet};

// // This would live in a networking crate
// pub trait PeerNetwork {
//     /// Request blocks by hash from peers
//     fn request_blocks(&mut self, hashes: Vec<Hash>);

//     /// Request headers from a specific height
//     fn request_headers(&mut self, from_height: u64);
// }

// impl BlockChain {
//     /// Call this periodically to sync with network
//     pub fn sync_with_peers<N: PeerNetwork>(&mut self, network: &mut N) {
//         // Get blocks we're missing
//         let missing = self.get_missing_blocks();

//         if !missing.is_empty() {
//             network.request_blocks(missing);
//         }

//         // Periodic maintenance
//         self.maintain();
//     }
// }
