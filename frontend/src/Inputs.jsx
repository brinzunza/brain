import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import './App.css'

const API_URL = 'http://localhost:8000'

function Inputs() {
  const [inputs, setInputs] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [deleteStatus, setDeleteStatus] = useState({})

  useEffect(() => {
    fetchInputs()
  }, [])

  const fetchInputs = async () => {
    setLoading(true)
    setError('')
    try {
      const response = await fetch(`${API_URL}/api/inputs`)
      const data = await response.json()

      if (data.status === 'success') {
        setInputs(data.inputs)
      } else {
        setError('error: ' + data.message)
      }
    } catch (err) {
      setError('error: ' + err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (inputId) => {
    setDeleteStatus({ [inputId]: 'deleting...' })
    try {
      const response = await fetch(`${API_URL}/api/inputs/${inputId}`, {
        method: 'DELETE'
      })

      const data = await response.json()

      if (data.status === 'success') {
        setDeleteStatus({ [inputId]: 'deleted' })
        // remove the input from the list
        setInputs(inputs.filter(input => input.id !== inputId))
        setTimeout(() => {
          setDeleteStatus({})
        }, 2000)
      } else {
        setDeleteStatus({ [inputId]: 'error: ' + data.message })
      }
    } catch (err) {
      setDeleteStatus({ [inputId]: 'error: ' + err.message })
    }
  }

  const formatDate = (timestamp) => {
    if (!timestamp) return 'unknown'
    try {
      return new Date(timestamp).toLocaleString()
    } catch {
      return 'unknown'
    }
  }

  const openaiInputs = inputs.filter(input => input.provider === 'openai')
  const ollamaInputs = inputs.filter(input => input.provider === 'ollama')

  const renderInputsList = (inputsList, providerName) => {
    if (inputsList.length === 0) {
      return (
        <div className="empty-state-small">
          no {providerName} inputs yet
        </div>
      )
    }

    return (
      <div className="inputs-list">
        {inputsList.map((input) => (
          <div key={input.id} className="input-item">
            <div className="input-header">
              <span className="input-type">
                {input.metadata.type === 'file'
                  ? `file: ${input.metadata.filename}`
                  : 'text'}
              </span>
              <span className="input-date">
                {formatDate(input.metadata.timestamp)}
              </span>
            </div>
            <div className="input-content">
              {input.content}
            </div>
            <div className="input-actions">
              <button
                className="button delete-button"
                onClick={() => handleDelete(input.id)}
                disabled={deleteStatus[input.id]}
              >
                {deleteStatus[input.id] || 'delete'}
              </button>
            </div>
          </div>
        ))}
      </div>
    )
  }

  return (
    <div className="app">
      <header className="header">
        <h1>brain - inputs</h1>
        <Link to="/" className="nav-link">back to main</Link>
      </header>

      {loading && <div className="status-message">loading...</div>}
      {error && <div className="status-message error">{error}</div>}

      {!loading && !error && inputs.length === 0 && (
        <div className="empty-state">
          no inputs stored yet. add some text or files from the main page.
        </div>
      )}

      {!loading && !error && inputs.length > 0 && (
        <div className="inputs-columns">
          <div className="inputs-column">
            <div className="column-header">
              <span className="column-title">openai embeddings</span>
              <span className="column-count">({openaiInputs.length})</span>
            </div>
            {renderInputsList(openaiInputs, 'openai')}
          </div>

          <div className="inputs-column">
            <div className="column-header">
              <span className="column-title">ollama embeddings</span>
              <span className="column-count">({ollamaInputs.length})</span>
            </div>
            {renderInputsList(ollamaInputs, 'ollama')}
          </div>
        </div>
      )}
    </div>
  )
}

export default Inputs
