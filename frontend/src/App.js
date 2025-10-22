import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar, Pie } from 'react-chartjs-2';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

// Base URL for the API
const API_URL = 'http://127.0.0.1:8000';

function App() {
  const [query, setQuery] = useState('');
  
  const [messages, setMessages] = useState([
    {
      sender: 'bot',
      text: "Welcome to your Kearney AI Procurement Analyst. I've loaded the `Sugar_Spend_Data.csv` file.\n\nYou can start by asking:\n• What is the total spend?\n• Which supplier has the highest spend?\n• Plot the spend for each commodity.",
      chart: null
    }
  ]);
  
  const [isLoading, setIsLoading] = useState(false);
  const [rawData, setRawData] = useState([]);
  const [isPanelOpen, setIsPanelOpen] = useState(false); 

  useEffect(() => {
    const fetchRawData = async () => {
      try {
        const response = await axios.get(`${API_URL}/api/data`);
        setRawData(response.data);
      } catch (error) {
        console.error("Error fetching raw data:", error);
      }
    };
    fetchRawData();
  }, []); 

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userMessage = { sender: 'user', text: query };
    const newMessages = [...messages, userMessage];

    setMessages(newMessages);
    setIsLoading(true);
    setQuery('');

    try {
      const response = await axios.post(`${API_URL}/api/chat`, {
        messages: newMessages.map((msg) => ({
          sender: msg.sender,
          text: msg.text,
        })),
      });

      const botMessage = {
        sender: 'bot',
        text: response.data.answer,
        chart: response.data.chart || null,
      };
      setMessages((prevMessages) => [...prevMessages, botMessage]);
    } catch (error) {
      console.error('Error fetching answer:', error);
      const errorMessage = {
        sender: 'bot',
        text: 'Sorry, something went wrong. Please check the console.',
        chart: null,
      };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Kearney AI Chatbot</h1>
        <p>Ask questions about your commodity data</p>
      </header>

      <div className="app-body">
        
        {/* Data Panel Component */}
        <DataPanel data={rawData} isOpen={isPanelOpen} />
        
        {/* Toggle Button */}
        <button
          onClick={() => setIsPanelOpen(!isPanelOpen)}
          className="panel-toggle"
          title={isPanelOpen ? "Hide Data Panel" : "Show Data Panel"}
        >
          {isPanelOpen ? '❮' : 'Data'}
        </button>

        {/* Chat Window */}
        <div className="chat-window">
          <div className="message-list">
            {messages.map((msg, index) => (
              <ChatMessage key={index} msg={msg} />
            ))}
            {isLoading && (
              <div className="message bot">
                <div className="loading-spinner"></div>
              </div>
            )}
          </div>
          <form className="chat-form" onSubmit={handleSubmit}>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g., Plot the spend for each commodity"
              disabled={isLoading}
            />
            <button type="submit" disabled={isLoading}>
              Send
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

// --- DataPanel Component ---
const DataPanel = ({ data, isOpen }) => {
  if (!data.length) {
    return (
      <div className={`data-panel ${isOpen ? 'open' : ''}`}>
        <div className="loading-spinner"></div>
      </div>
    );
  }

  const headers = Object.keys(data[0]);

  return (
    <div className={`data-panel ${isOpen ? 'open' : ''}`}>
      <h3>Sugar Spend Data</h3>
      <div className="table-container">
        <table>
          <thead>
            <tr>
              {headers.map(header => <th key={header}>{header}</th>)}
            </tr>
          </thead>
          <tbody>
            {data.map((row, index) => (
              <tr key={index}>
                {headers.map(header => <td key={row[header] + index}>{row[header]}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};


// --- Chart.js options ---
const chartOptions = {
  bar: {
    responsive: true,
    plugins: {
      legend: { position: 'top', labels: { color: '#333' } },
      title: { display: true, text: 'Spend Analysis', color: '#000033', font: { size: 16 } },
    },
    scales: {
      y: { ticks: { color: '#333' } },
      x: { ticks: { color: '#333' } },
    },
  },
  pie: {
    responsive: true,
    plugins: {
      legend: { position: 'top', labels: { color: '#333' } },
      title: { display: true, text: 'Spend Distribution', color: '#000033', font: { size: 16 } },
    },
  },
};

// --- ChatMessage component ---
const ChatMessage = ({ msg }) => {
  const canRenderChart =
    msg.sender === 'bot' &&
    msg.chart &&
    msg.chart.labels &&
    msg.chart.data;

  if (canRenderChart) {
    const chartData = {
      labels: msg.chart.labels,
      datasets: [
        {
          label: 'Spend (USD)',
          data: msg.chart.data,
          backgroundColor: [
            'rgba(0, 0, 51, 0.7)',
            'rgba(0, 122, 255, 0.7)',
            'rgba(52, 199, 89, 0.7)',
            'rgba(255, 149, 0, 0.7)',
            'rgba(88, 86, 214, 0.7)',
            'rgba(255, 45, 85, 0.7)',
            'rgba(175, 82, 222, 0.7)',
            'rgba(255, 59, 48, 0.7)',
            'rgba(90, 200, 250, 0.7)',
            'rgba(255, 204, 0, 0.7)',
          ],
          borderColor: [
            'rgba(0, 0, 51, 1)',
            'rgba(0, 122, 255, 1)',
            'rgba(52, 199, 89, 1)',
            'rgba(255, 149, 0, 1)',
            'rgba(88, 86, 214, 1)',
            'rgba(255, 45, 85, 1)',
            'rgba(175, 82, 222, 1)',
            'rgba(255, 59, 48, 1)',
            'rgba(90, 200, 250, 1)',
            'rgba(255, 204, 0, 1)',
          ],
          borderWidth: 1,
        },
      ],
    };

    return (
      <div className="message bot">
        {msg.text && <pre>{msg.text}</pre>}
        <div className="chart-container">
          {msg.chart.type === 'bar' && <Bar data={chartData} options={chartOptions.bar} />}
          {msg.chart.type === 'pie' && <Pie data={chartData} options={chartOptions.pie} />}
        </div>
      </div>
    );
  }

  return (
    <div className={`message ${msg.sender}`}>
      <pre>{msg.text}</pre>
    </div>
  );
};

export default App;

