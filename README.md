# P2P Blockchain

A peer-to-peer blockchain built from scratch in Rust. This was a learning project to better understand how blockchains work by recreating a P2P network where nodes discover each other, share transactions, mine blocks, and maintain consensus.

## Overview

Nodes connect over a local network using [libp2p](https://libp2p.io/), automatically discover peers via mDNS, and broadcast blocks/transactions with Gossipsub. Each node maintains its own copy of the chain, validates incoming blocks, handles forks, and tracks peer reputation.

### Key features

- **Proof-of-work mining** with configurable difficulty (BLAKE3 hashing)
- **Ed25519 wallets** for signing and verifying transactions
- **Peer discovery** via mDNS — no manual configuration needed
- **Gossipsub** for broadcasting blocks and transactions
- **Request-response** protocol for syncing chain state between peers
- **Fork handling** with cumulative difficulty tracking
- **Orphan block pool** for out-of-order block arrival
- **Peer reputation** system that bans nodes sending invalid data
- **Interactive CLI** for sending coins, mining, and checking status

## Project Structure

```
crates/
├── blockchain-types/   Core types: blocks, transactions, wallets, mining
├── blockchain/         Chain management, fork handling, orphan pool
├── mempool/            Pending transaction queue
├── network/            libp2p networking (Gossipsub + mDNS + request-response)
├── node/               Orchestrates blockchain, network, mempool, and sync
└── cli/                Interactive terminal interface
```

## Getting Started

### Prerequisites

- [Rust](https://rustup.rs/) (nightly toolchain — automatically selected via `rust-toolchain.toml`)

### Build

```bash
cargo build --release
```

### Run

```bash
# Generate a random wallet
cargo run --release -p cli

# Or provide a seed for a deterministic wallet
cargo run --release -p cli -- myseed
```

To run multiple nodes on the same machine, open separate terminals and run the command in each — they will discover each other automatically.

### CLI Commands

| Command | Description |
|---|---|
| `balance` | Show your current balance |
| `send <address> <amount>` | Send coins to another address |
| `mine` | Mine a new block |
| `status` | Show chain height, difficulty, peers, and mempool size |
| `quit` | Exit |

Logs are written to `blockchain.log`. Use `tail -f blockchain.log` in another terminal to watch activity.

## Dependencies

| Crate | Purpose |
|---|---|
| `libp2p` | P2P networking (Gossipsub, mDNS, request-response) |
| `blake3` | Block and transaction hashing |
| `ed25519-dalek` | Transaction signatures |
| `tokio` | Async runtime |
| `serde` | Serialization |
| `rayon` | Parallel mining |
| `rustyline` | CLI input |
