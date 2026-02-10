import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import './App.css'

const API_URL = 'http://localhost:8000'

function Inputs() {
  const [inputs, setInputs] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [deleteStatus, setDeleteStatus] = useState({})
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] })
  const [graphLoading, setGraphLoading] = useState(true)
  const [showGraph, setShowGraph] = useState(true)

  useEffect(() => {
    fetchInputs()
    fetchGraphData()
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

  const fetchGraphData = async () => {
    setGraphLoading(true)
    try {
      const response = await fetch(`${API_URL}/api/graph/visualization?limit=50`)
      const data = await response.json()

      if (data.available) {
        setGraphData({
          nodes: data.nodes || [],
          edges: data.edges || [],
          stats: data.stats || {}
        })
      }
    } catch (err) {
      console.error('Error fetching graph data:', err)
    } finally {
      setGraphLoading(false)
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
        setInputs(inputs.filter(input => input.id !== inputId))
        // Refresh graph data after deletion
        fetchGraphData()
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

  const getNodeColor = (type) => {
    const colors = {
      'technology': '#00ff88',
      'organization': '#ff8800',
      'location': '#8800ff',
      'concept': '#00ccff',
      'person': '#ff0088',
      'unknown': '#666666'
    }
    return colors[type] || colors['unknown']
  }

  return (
    <div className="app">
      <header className="header">
        <h1>brain - stored inputs</h1>
        <Link to="/" className="nav-link">back to main</Link>
      </header>

      {loading && <div className="status-message">loading...</div>}
      {error && <div className="status-message error">{error}</div>}

      {!loading && !error && inputs.length === 0 && (
        <div className="empty-state">
          no inputs stored yet. add some text from the main page.
        </div>
      )}

      {!loading && !error && inputs.length > 0 && (
        <>
          {/* Knowledge Graph Section */}
          <div className="section">
            <div className="section-header">
              <div className="section-title">
                knowledge graph
                {graphData.stats && (
                  <span className="graph-stats">
                    {graphData.stats.node_count} entities · {graphData.stats.edge_count} connections
                  </span>
                )}
              </div>
              <button
                className="button toggle-button"
                onClick={() => setShowGraph(!showGraph)}
              >
                {showGraph ? 'hide' : 'show'}
              </button>
            </div>

            {showGraph && (
              <>
                {graphLoading && <div className="status-message">loading graph...</div>}

                {!graphLoading && graphData.nodes.length === 0 && (
                  <div className="empty-state">
                    no entities extracted yet. the graph will appear after adding documents with recognizable entities.
                  </div>
                )}

                {!graphLoading && graphData.nodes.length > 0 && (
                  <div className="graph-container">
                    <div className="graph-legend">
                      <span className="legend-item">
                        <span className="legend-dot" style={{background: '#00ff88'}}></span> technology
                      </span>
                      <span className="legend-item">
                        <span className="legend-dot" style={{background: '#ff8800'}}></span> organization
                      </span>
                      <span className="legend-item">
                        <span className="legend-dot" style={{background: '#8800ff'}}></span> location
                      </span>
                      <span className="legend-item">
                        <span className="legend-dot" style={{background: '#00ccff'}}></span> concept
                      </span>
                      <span className="legend-item">
                        <span className="legend-dot" style={{background: '#ff0088'}}></span> person
                      </span>
                    </div>

                    <div className="graph-visualization">
                      <svg width="100%" height="500" style={{background: '#0a0a0a', border: '1px solid #333'}}>
                        {/* Draw edges first (so they appear behind nodes) */}
                        {graphData.edges.map((edge, idx) => {
                          const fromNode = graphData.nodes.find(n => n.id === edge.from)
                          const toNode = graphData.nodes.find(n => n.id === edge.to)
                          if (!fromNode || !toNode) return null

                          // Simple layout: circular arrangement
                          const total = graphData.nodes.length
                          const fromAngle = (graphData.nodes.indexOf(fromNode) / total) * 2 * Math.PI
                          const toAngle = (graphData.nodes.indexOf(toNode) / total) * 2 * Math.PI
                          const radius = Math.min(350, 150 + total * 3)
                          const centerX = 400
                          const centerY = 250

                          const x1 = centerX + radius * Math.cos(fromAngle)
                          const y1 = centerY + radius * Math.sin(fromAngle)
                          const x2 = centerX + radius * Math.cos(toAngle)
                          const y2 = centerY + radius * Math.sin(toAngle)

                          return (
                            <line
                              key={`edge-${idx}`}
                              x1={x1}
                              y1={y1}
                              x2={x2}
                              y2={y2}
                              stroke="#333"
                              strokeWidth={Math.min(edge.weight || 1, 3)}
                              opacity="0.5"
                            />
                          )
                        })}

                        {/* Draw nodes */}
                        {graphData.nodes.map((node, idx) => {
                          const total = graphData.nodes.length
                          const angle = (idx / total) * 2 * Math.PI
                          const radius = Math.min(350, 150 + total * 3)
                          const centerX = 400
                          const centerY = 250
                          const x = centerX + radius * Math.cos(angle)
                          const y = centerY + radius * Math.sin(angle)

                          return (
                            <g key={`node-${node.id}`}>
                              <circle
                                cx={x}
                                cy={y}
                                r="8"
                                fill={getNodeColor(node.type)}
                                stroke="#000"
                                strokeWidth="2"
                              />
                              <text
                                x={x}
                                y={y - 15}
                                textAnchor="middle"
                                fill={getNodeColor(node.type)}
                                fontSize="11"
                                fontFamily="'JetBrains Mono', monospace"
                              >
                                {node.label.length > 15 ? node.label.substring(0, 15) + '...' : node.label}
                              </text>
                            </g>
                          )
                        })}
                      </svg>
                    </div>

                    <div className="graph-info">
                      showing connections between entities that appear together in your documents
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          {/* Documents Section */}
          <div className="section">
            <div className="section-title">
              stored documents ({inputs.length})
            </div>
            <div className="inputs-list">
              {inputs.map((input) => (
                <div key={input.id} className="input-item">
                  <div className="input-header">
                    <span className="input-id">
                      {input.id.substring(0, 16)}...
                    </span>
                    <div className="input-metadata">
                      <span className="chunk-count">
                        {input.chunk_count || 1} chunk{input.chunk_count !== 1 ? 's' : ''}
                      </span>
                      {input.token_count > 0 && (
                        <span className="token-count-badge">
                          {input.token_count} tokens
                        </span>
                      )}
                    </div>
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
          </div>
        </>
      )}
    </div>
  )
}

export default Inputs
