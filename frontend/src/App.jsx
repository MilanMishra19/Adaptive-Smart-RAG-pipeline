import { useState, useRef, useEffect } from 'react';
import './index.css';
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL;
const App = () => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [isThinking, setIsThinking] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsThinking(true);

    try {
      // PLUG YOUR BACKEND HERE
        const response = await fetch(BACKEND_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: input})
       });
       const data = await response.json();
      
      // Simulated delay for UI feel
      setMessages(prev => [...prev, {
            role: 'ai',
            content: data.answer,
            sources: data.sources?.map(s => `Paper ${s.paper_id} — ${s.section_title}`) ?? []
    }]);
    setIsThinking(false);

    } catch (error) {
      console.error("Uplink Error:", error);
      setIsThinking(false);
    }
  };

  return (
    <div className="app-container">
      {/* Background Decor */}
      <div className="ambient-glow glow-left"></div>
      <div className="ambient-glow glow-right"></div>

      {/* Messages */}
      <div className="chat-window" ref={scrollRef}>
        {messages.length === 0 && (
          <div style={{ textAlign: 'center', marginTop: '20vh', color: 'var(--muted-text)' }}>
            <h2 style={{ color: '#fff', marginBottom: '8px' }}>Intelligence Reimagined</h2>
            <p>Select a mode and start your query.</p>
          </div>
        )}
        
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role === 'user' ? 'user-message' : 'ai-message'}`}>
            <p>{msg.content}</p>
            {msg.sources && (
              <div className="sources">
                {msg.sources.map((s, idx) => <span key={idx} className="source-pill">{s}</span>)}
              </div>
            )}
          </div>
        ))}
        {isThinking && (
          <div className="ai-message" style={{ opacity: 0.5 }}>AI is processing knowledge...</div>
        )}
      </div>

      {/* Input Section */}
      <div className="interaction-hub">
        <div className="input-pill">
          <input 
            type="text" 
            placeholder="Ask anything..." 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          />

          <button className="send-btn" onClick={handleSend}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
};

export default App;