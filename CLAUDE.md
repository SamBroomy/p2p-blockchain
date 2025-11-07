# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A peer-to-peer blockchain implementation in Rust with cryptographic transactions, proof-of-work mining, and fork resolution. The project uses a workspace structure with a `domain` crate containing core blockchain logic.

## Architecture

### Core Domain Model (`crates/domain/`)

**Block Construction & Mining**

- `Block`: Immutable, validated blocks with proof-of-work
- `BlockConstructor`: Builder pattern for mining new blocks
- Mining uses blake3 hashing with configurable difficulty (leading zero bits)
- Blocks contain transactions, previous hash, nonce, and timestamp
- Invariant: `Block` instances are always valid (enforced via constructor and serde validation)

**Blockchain & Fork Resolution**

- `BlockChain`: Main chain plus fork tracking with automatic reorganization
- `Chain<T>`: Generic chain type supporting both `RootChain` and `ForkChain`
  - `RootChain`: Connects to genesis, tracks absolute account balances
  - `ForkChain`: Branches from root or other forks, validated against parent
- Fork resolution based on cumulative difficulty (sum of 2^difficulty for all blocks)
- Automatic chain reorganization when fork exceeds main chain difficulty
- Forks older than 10 blocks from main chain tip are pruned

**Transactions & Wallets**

- `Transaction`: Signed transfers using ed25519 signatures
- `TransactionConstructor`: Creates and signs transactions
- `Wallet`: Key management (random or deterministic from seed)
- Transactions validated for signature correctness and hash integrity
- Invariant: `Transaction` instances are always valid (enforced via constructor and serde validation)

**Account State**

- Simple UTXO-like balance tracking
- Initial balance of 100 for new accounts (no block rewards yet)
- Balance validation during block addition prevents negative balances

**Mempool**

- Transaction queue with deduplication by hash
- Simple FIFO ordering (no fee-based prioritization)

## Development Commands

### Building

```bash
cargo build              # Build workspace
cargo build -p domain    # Build domain crate only
cargo build --release    # Release build
```

### Testing

```bash
cargo test              # Run all tests
cargo test -p domain    # Test domain crate only
cargo test -q           # Quiet mode (one line per test)
cargo test test_name    # Run specific test
```

### Linting

```bash
cargo clippy            # Run clippy with workspace lints
cargo clippy --fix      # Auto-fix issues where possible
cargo fmt              # Format code
cargo fmt --check      # Check formatting without modifying
```

### Running

```bash
cargo run              # Run main binary (currently empty/commented)
```

## Clippy Configuration

Strict linting with workspace-level configuration in `Cargo.toml`:

- `dbg_macro`, `todo`, `unimplemented` are **denied** (compilation errors)
- Pedantic lints enabled but some relaxed (`module_name_repetitions`, `too_many_arguments`, etc.)
- `unsafe_code` is **forbidden** at workspace level
- `unwrap_used` is allowed (not denied)

## Key Implementation Details

### Hash Validation

Both blocks and transactions use two-phase hashing:

1. Hash inner data (transactions/amounts/keys)
2. Hash complete structure including nonce/signature
This allows efficient mining (clone partial hash state) and signature verification.

### Chain Validation

- Structural: Each block's `previous_hash` matches prior block's `hash`
- Difficulty: Block hash meets target (correct number of leading zeros)
- Balances: All transactions in chain maintain non-negative account balances
- Fork validity: Fork must connect to root chain and maintain valid balances from fork point

### Type Safety via Generics

The `Chain<T>` type uses sealed trait bounds to enforce:

- `RootChain`: Must connect to genesis, fully validated balances
- `ForkChain`: May branch from any point, validated against parent chain
This prevents invalid chain state transitions at compile time.

### Serde with Validation

Both `Block` and `Transaction` use `serde_valid` to maintain invariants:

- Custom validators ensure deserialized blocks/transactions are cryptographically valid
- Prevents creating invalid domain objects even from untrusted input
- See `#[validate(custom = ...)]` attributes on structs

## Testing

Tests are colocated in `#[cfg(test)]` modules within each source file:

- `block.rs`: Mining, serialization, transaction inclusion
- `transaction.rs`: Signing, validation, roundtrip serialization
- `wallet.rs`: Key generation, deterministic seeds, transaction creation

No integration test directory currently exists.
