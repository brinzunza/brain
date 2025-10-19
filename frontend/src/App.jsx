import { useState } from 'react'
import { Link } from 'react-router-dom'
import './App.css'

const API_URL = 'http://localhost:8000'

function App() {
  const [textInput, setTextInput] = useState('')
  const [file, setFile] = useState(null)
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState([])
  const [textStatus, setTextStatus] = useState('')
  const [fileStatus, setFileStatus] = useState('')
  const [loading, setLoading] = useState(false)
  const [showClearConfirm, setShowClearConfirm] = useState(false)
  const [clearStatus, setClearStatus] = useState('')
  const [llmProvider, setLlmProvider] = useState('chatgpt')
  const [embeddingProvider, setEmbeddingProvider] = useState('openai')

  const handleTextSubmit = async () => {
    if (!textInput.trim()) return

    setTextStatus('storing...')
    try {
      const response = await fetch(`${API_URL}/api/add-text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: textInput.toLowerCase(),
          embedding_provider: embeddingProvider
        })
      })

      const data = await response.json()
      if (data.status === 'success') {
        setTextStatus('stored')
        setTextInput('')
        setTimeout(() => setTextStatus(''), 2000)
      } else {
        setTextStatus('error: ' + data.message)
      }
    } catch (error) {
      setTextStatus('error: ' + error.message)
    }
  }

  const handleFileSubmit = async () => {
    if (!file) return

    setFileStatus('storing...')
    const formData = new FormData()
    formData.append('file', file)
    formData.append('embedding_provider', embeddingProvider)

    try {
      const response = await fetch(`${API_URL}/api/add-file`, {
        method: 'POST',
        body: formData
      })

      const data = await response.json()
      if (data.status === 'success') {
        setFileStatus('stored')
        setFile(null)
        setTimeout(() => setFileStatus(''), 2000)
      } else {
        setFileStatus('error: ' + data.message)
      }
    } catch (error) {
      setFileStatus('error: ' + error.message)
    }
  }

  const handleAsk = async () => {
    if (!question.trim()) return

    const userQuestion = question
    setQuestion('')
    setMessages(prev => [...prev, { type: 'question', content: userQuestion }])
    setLoading(true)

    try {
      const response = await fetch(`${API_URL}/api/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: userQuestion,
          llm_provider: llmProvider,
          embedding_provider: embeddingProvider
        })
      })

      const data = await response.json()
      if (data.status === 'success') {
        setMessages(prev => [...prev, { type: 'answer', content: data.answer }])
      } else {
        setMessages(prev => [...prev, { type: 'error', content: 'error: ' + data.message }])
      }
    } catch (error) {
      setMessages(prev => [...prev, { type: 'error', content: 'error: ' + error.message }])
    } finally {
      setLoading(false)
    }
  }

  const handleClearBrain = async () => {
    setClearStatus('clearing...')
    try {
      const response = await fetch(`${API_URL}/api/clear`, {
        method: 'DELETE'
      })

      const data = await response.json()
      if (data.status === 'success') {
        setClearStatus('cleared')
        setShowClearConfirm(false)
        setMessages([])
        setTimeout(() => setClearStatus(''), 2000)
      } else {
        setClearStatus('error: ' + data.message)
      }
    } catch (error) {
      setClearStatus('error: ' + error.message)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>brain</h1>
        <div className="header-controls">
          <button
            className="provider-toggle-button"
            onClick={() => setEmbeddingProvider(prev => prev === 'openai' ? 'ollama' : 'openai')}
          >
            embeddings: {embeddingProvider}
          </button>
          <Link to="/inputs" className="nav-link">inputs</Link>
        </div>
      </header>

      <div className="section">
        <div className="section-title">add text</div>
        <textarea
          value={textInput}
          onChange={(e) => setTextInput(e.target.value.toLowerCase())}
          placeholder="type anything you want to remember..."
        />
        <button className="button" onClick={handleTextSubmit}>
          store
        </button>
        {textStatus && (
          <div className={`status-message ${textStatus === 'stored' ? 'success' : textStatus.startsWith('error') ? 'error' : ''}`}>
            {textStatus}
          </div>
        )}
      </div>

      <div className="section">
        <div className="section-title">add file</div>
        <div className="file-input-wrapper">
          <label htmlFor="file-input" className="file-input-label">
            choose file
          </label>
          <input
            id="file-input"
            type="file"
            onChange={(e) => setFile(e.target.files[0])}
          />
          {file && <span className="file-name">{file.name}</span>}
        </div>
        <div></div>
        <button className="button" onClick={handleFileSubmit} disabled={!file}>
          store
        </button>
        {fileStatus && (
          <div className={`status-message ${fileStatus === 'stored' ? 'success' : fileStatus.startsWith('error') ? 'error' : ''}`}>
            {fileStatus}
          </div>
        )}
      </div>

      <div className="section">
        <div className="section-title">
          ask questions
          <button
            className="llm-toggle-button"
            onClick={() => setLlmProvider(prev => prev === 'chatgpt' ? 'ollama' : 'chatgpt')}
          >
            {llmProvider === 'chatgpt' ? 'chatgpt' : 'ollama'}
          </button>
        </div>
        {messages.length > 0 && (
          <div className="messages">
            {messages.map((msg, idx) => (
              <div key={idx} className="message">
                <div className="message-label">
                  {msg.type === 'question' ? 'you' : msg.type === 'error' ? 'error' : 'brain'}
                </div>
                <div className="message-content">{msg.content}</div>
              </div>
            ))}
          </div>
        )}
        <div className="question-input">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value.toLowerCase())}
            placeholder="ask anything..."
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                handleAsk()
              }
            }}
          />
          <button className="button" onClick={handleAsk} disabled={loading}>
            {loading ? 'thinking...' : 'ask'}
          </button>
        </div>
      </div>

      <div className="section">
        <div className="section-title danger">clear brain</div>
        {!showClearConfirm ? (
          <button
            className="button danger"
            onClick={() => setShowClearConfirm(true)}
          >
            clear all data
          </button>
        ) : (
          <div className="clear-confirm">
            <div className="warning-text">
              this will permanently delete all stored data. are you sure?
            </div>
            <div className="clear-actions">
              <button className="button danger" onClick={handleClearBrain}>
                yes, clear everything
              </button>
              <button className="button" onClick={() => setShowClearConfirm(false)}>
                cancel
              </button>
            </div>
          </div>
        )}
        {clearStatus && (
          <div className={`status-message ${clearStatus === 'cleared' ? 'success' : clearStatus.startsWith('error') ? 'error' : ''}`}>
            {clearStatus}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
