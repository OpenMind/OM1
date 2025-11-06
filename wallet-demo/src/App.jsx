import React, { useState, useEffect } from "react";
import { createWalletClient, custom, parseEther, formatEther } from "viem";
import { sepolia, baseSepolia, lineaTestnet } from "viem/chains";
import {
  useAccount,
  useConnect,
  useDisconnect,
  useSignMessage,
  useSwitchNetwork,
  useBalance,
} from "wagmi";
import { MetaMaskConnector } from "wagmi/connectors/metaMask";
import { WalletConnectConnector } from "wagmi/connectors/walletConnect";
import { CoinbaseWalletConnector } from "wagmi/connectors/coinbaseWallet";
import { InjectedConnector } from "wagmi/connectors/injected";
import { createConfig, WagmiConfig, configureChains } from "wagmi";
import { publicProvider } from "wagmi/providers/public";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "./App.css";

const projectId = "93aa395edb75ceaf138f36d3ca999af2";
const queryClient = new QueryClient();

const { chains, publicClient } = configureChains(
  [sepolia, baseSepolia, lineaTestnet],
  [publicProvider()]
);

const config = createConfig({
  autoConnect: true,
  connectors: [
    new MetaMaskConnector({ chains }),
    new InjectedConnector({
      chains,
      options: {
        name: "Injected Wallet (Rabby / Zerion / Trust)",
        shimDisconnect: true,
      },
    }),
    new WalletConnectConnector({ chains, options: { projectId } }),
    new CoinbaseWalletConnector({
      chains,
      options: { appName: "OM1 Wallet Demo" },
    }),
  ],
  publicClient,
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <WagmiConfig config={config}>
        <WalletApp />
      </WagmiConfig>
    </QueryClientProvider>
  );
}

function WalletApp() {
  const { address, isConnected, connector, chain } = useAccount();
  const { connect, connectors } = useConnect();
  const { disconnect } = useDisconnect();
  const { signMessageAsync } = useSignMessage();
  const { switchNetwork } = useSwitchNetwork();
  const { data: balanceData, refetch: refetchBalance } = useBalance({
    address,
    watch: true,
  });

  const [selectedChain, setSelectedChain] = useState("sepolia");
  const [recipient, setRecipient] = useState("");
  const [amount, setAmount] = useState("");
  const [transactions, setTransactions] = useState([]);
  const [isSigned, setIsSigned] = useState(false);

  useEffect(() => {
    if (isConnected) {
      refetchBalance();
    }
  }, [isConnected, chain, refetchBalance]);

  const handleSign = async () => {
    try {
      const message = "OM1 Wallet Signature Test ✅";
      const signature = await signMessageAsync({ message });
      alert("Signature complete ✅\n" + signature.substring(0, 20) + "...");
      setIsSigned(true);
    } catch {
      alert("Signature cancelled ❌");
    }
  };

  const handleSend = async () => {
    if (!recipient || !amount) {
      alert("Please enter recipient and amount.");
      return;
    }

    try {
      const [account] = await window.ethereum.request({
        method: "eth_requestAccounts",
      });

      const client = createWalletClient({
        account,
        chain:
          selectedChain === "baseSepolia"
            ? baseSepolia
            : selectedChain === "linea"
            ? lineaTestnet
            : sepolia,
        transport: custom(window.ethereum),
      });

      const hash = await client.sendTransaction({
        to: recipient,
        value: parseEther(amount),
      });

      alert("Transaction sent ✅\nHash: " + hash.substring(0, 15) + "...");
      setTransactions((prev) => [{ hash, chain: selectedChain }, ...prev.slice(0, 4)]);
      refetchBalance();
    } catch (err) {
      alert("Transaction failed ❌\n" + err.message);
    }
  };

  const handleSwitch = (network) => {
    setSelectedChain(network);
    switchNetwork(
      network === "baseSepolia"
        ? 84532
        : network === "linea"
        ? 59141
        : 11155111
    );
  };

  const getChainName = () =>
    chain?.name ||
    (selectedChain === "baseSepolia"
      ? "Base Sepolia Testnet"
      : selectedChain === "linea"
      ? "Linea Goerli Testnet"
      : "Sepolia Testnet");

  return (
    <div className="app-container">
      <div className="wallet-section">
        <div className="header">
          <img src="/openmind-logo.png" alt="OpenMind Logo" className="logo" />
          <h2 className="brand-text">OPENMIND</h2>
        </div>

        <h1 className="title">✅ OM1 Wallet is Live</h1>

        {!isConnected ? (
          <>
            <p>Connect MetaMask, WalletConnect, Coinbase, or Injected Wallet</p>
            {connectors.map((connector) => (
              <button
                key={connector.id}
                onClick={() => connect({ connector })}
                className="connect-btn"
              >
                {connector.name}
              </button>
            ))}
          </>
        ) : (
          <>
            <p>
              Connected:{" "}
              <span className="address">{address}</span> via{" "}
              {connector?.name || "Unknown"}
            </p>
            <p>Network: {getChainName()}</p>
            <p className="balance">
              💰 Balance:{" "}
              {balanceData
                ? `${parseFloat(formatEther(balanceData.value)).toFixed(4)} ETH`
                : "Loading..."}
            </p>

            {!isSigned ? (
              <button onClick={handleSign} className="sign-btn">
                Sign Message
              </button>
            ) : (
              <p className="signed-text">✅ Message Signed Successfully</p>
            )}

            <div className="send-box">
              <input
                type="text"
                placeholder="Recipient address"
                value={recipient}
                onChange={(e) => setRecipient(e.target.value)}
              />
              <input
                type="text"
                placeholder="Amount (ETH)"
                value={amount}
                onChange={(e) => setAmount(e.target.value)}
              />
              <button onClick={handleSend} className="send-btn">
                Send ETH
              </button>
            </div>

            <div className="switch-buttons">
              <button onClick={() => handleSwitch("sepolia")}>
                Switch to Sepolia
              </button>
              <button onClick={() => handleSwitch("baseSepolia")}>
                Switch to Base Sepolia
              </button>
              <button onClick={() => handleSwitch("linea")}>
                Switch to Linea Goerli
              </button>
            </div>

            <button onClick={disconnect} className="disconnect-btn">
              Disconnect
            </button>
          </>
        )}
      </div>

      <div className="tx-panel">
        <h3>🧾 Recent Transactions</h3>
        {transactions.length === 0 ? (
          <p>No transactions yet.</p>
        ) : (
          <ul>
            {transactions.map((tx, idx) => (
              <li key={idx}>
                <a
                  href={`https://sepolia.etherscan.io/tx/${tx.hash}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {tx.hash.substring(0, 22)}... ({tx.chain})
                </a>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
