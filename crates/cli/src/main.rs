use std::{error::Error, time::Duration};

use blockchain_types::wallet::{Address, Wallet};
use futures::StreamExt;
use node::Node;
use rustyline::{DefaultEditor, error::ReadlineError};

struct App {
    input: String,
    messages: Vec<String>,
    scroll: usize,
}

impl App {
    fn new() -> Self {
        Self {
            input: String::new(),
            messages: Vec::new(),
            scroll: 0,
        }
    }

    fn add_message(&mut self, msg: impl Into<String>) {
        self.messages.push(msg.into());
        // Keep last 100 messages
        if self.messages.len() > 100 {
            self.messages.remove(0);
        }
        // Auto-scroll to bottom
        self.scroll = self.messages.len().saturating_sub(1);
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Write logs to file to keep CLI clean
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("blockchain.log")?;
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                tracing_subscriber::EnvFilter::new("debug")
                    .add_directive("libp2p_mdns::behaviour::iface=off".parse().unwrap())
                    .add_directive("libp2p_mdns::behaviour=warn".parse().unwrap())
                    .add_directive("libp2p_gossipsub=warn".parse().unwrap())
                    // Show your app logs at debug level
                    .add_directive("cli=debug".parse().unwrap())
                    .add_directive("node=debug".parse().unwrap())
            }),
        )
        .with_writer(log_file)
        .compact()
        .init();
    let args: Vec<String> = std::env::args().collect();
    // Get wallet seed from args or generate random
    let seed = args.get(1).map(String::as_str);
    let wallet = seed.map_or_else(
        || {
            eprintln!("⚠️  No seed provided, generating random wallet");
            Wallet::new()
        },
        Wallet::from_seed,
    );

    let mut node = Node::new(wallet)?;

    // CLI output goes to stdout (using println!)
    println!("🔐 Your address: {}", node.my_address());
    println!("💰 Balance: {}", node.get_balance_info());
    println!();
    println!("Commands:");
    println!("  balance              - Show your balance");
    println!("  send <addr> <amount> - Send coins to an address");
    println!("  mine                 - Mine a block");
    println!("  status               - Show blockchain status");
    println!("  quit                 - Exit");
    println!();
    println!("💡 Tip: Logs are written to blockchain.log");
    println!("   Run 'tail -f blockchain.log' in another terminal to watch logs");

    // Create async channel for commands
    let (cmd_tx, mut cmd_rx) = tokio::sync::mpsc::channel::<String>(100);

    // Spawn readline in blocking thread
    std::thread::spawn(move || {
        let mut rl = DefaultEditor::new().unwrap();
        loop {
            match rl.readline(">> ") {
                Ok(line) => {
                    let _ = cmd_tx.blocking_send(line);
                }
                Err(ReadlineError::Interrupted | ReadlineError::Eof) => break,
                Err(err) => {
                    eprintln!("Error: {err:?}");
                    break;
                }
            }
        }
    });

    // Merge both event loops: node sync + CLI commands
    let mut sync_interval = tokio::time::interval(Duration::from_secs(10));

    loop {
        tokio::select! {
            _ = sync_interval.tick() => {
                node.sync_check();
            }

            event = node.swarm.select_next_some() => {
                node.handle_swarm_event(event);
            }

            Some(line) = cmd_rx.recv() => {
                let parts: Vec<&str> = line.split_whitespace().collect();
                match parts.as_slice() {
                    ["balance"] => {
                        println!("💰 Balance: {}", node.get_balance_info());
                    }

                    ["send", addr_str, amount_str] => {
                        let Ok(amount) = amount_str.parse::<u64>() else {
                            println!("❌ Invalid amount");
                            continue;
                        };
                        let Ok(address) = Address::from_hex(addr_str) else {
                            println!("❌ Invalid address");
                            continue;
                        };
                        match node.send_transaction(&address, amount) {
                            Ok(tx) => println!("✅ Sent {} coins (tx: {:?})", amount, tx.hash()),
                            Err(e) => println!("❌ Error: {e}"),
                        }
                    }

                    ["mine"] => {
                        println!("⛏️  Mining...");
                        match node.mine_block() {
                            Ok(block) => println!("✅ Mined block {:?}", block.hash()),
                            Err(e) => println!("❌ Error: {e}"),
                        }
                    }

                    ["status"] => {
                        println!("📊 Blockchain Status:");
                        println!("   Height: {}", node.blockchain.height());
                        println!("   Difficulty: {}", node.blockchain.difficulty);
                        println!("   Peers: {:#?}", node.list_peers().collect::<Vec<_>>());
                        println!("   Mempool: {} txs", node.mempool.len());
                        println!("   Orphans: {}", node.blockchain.orphan_count());
                    }

                    ["quit" | "exit"] => {
                        println!("👋 Goodbye!");
                        break;
                    }

                    [] => {
                        // Empty line, ignore
                    }

                    _ => {
                        println!("❌ Unknown command");
                    }
                }
            }
        }
    }

    Ok(())
}
