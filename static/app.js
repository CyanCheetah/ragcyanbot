document.addEventListener('DOMContentLoaded', function() {
    // UI Elements
    const chatHistory = document.getElementById('chatHistory');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const uploadStatus = document.getElementById('uploadStatus');
    const searchInput = document.getElementById('searchInput');
    const searchButton = document.getElementById('searchButton');
    const searchResults = document.getElementById('searchResults');
    const updateGraphBtn = document.getElementById('updateGraphBtn');
    const newChatBtn = document.getElementById('newChatBtn');
    const tabButtons = document.querySelectorAll('.tab-button');
    const sections = document.querySelectorAll('[data-section]');

    // Tab Navigation
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.dataset.tab;
            
            // Update active states
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // Show/hide sections
            sections.forEach(section => {
                if (section.dataset.section === targetTab) {
                    section.classList.add('active');
                } else {
                    section.classList.remove('active');
                }
            });
        });
    });

    // New Chat
    newChatBtn.addEventListener('click', () => {
        chatHistory.innerHTML = '';
        showSection('chat');
    });

    // Chat Functionality
    function addMessage(text, type) {
        const messageList = chatHistory.querySelector('.message-list') || chatHistory;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const iconDiv = document.createElement('div');
        iconDiv.className = 'message-icon';
        iconDiv.innerHTML = type === 'user' ? 
            '<i class="fas fa-user"></i>' : 
            '<i class="fas fa-robot"></i>';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = text;
        
        messageDiv.appendChild(iconDiv);
        messageDiv.appendChild(contentDiv);
        messageList.appendChild(messageDiv);
        
        // Scroll to the latest message
        messageDiv.scrollIntoView({ behavior: 'smooth' });
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        // Add user message
        addMessage(message, 'user');
        userInput.value = '';

        // Show loading state
        document.body.classList.add('loading');

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message })
            });

            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Failed to get response');
            }
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            addMessage(data.response, 'assistant');
        } catch (error) {
            console.error('Chat error:', error);
            addMessage(`I apologize, but I encountered an error: ${error.message}`, 'error');
        } finally {
            document.body.classList.remove('loading');
        }
    }

    // Upload Functionality
    async function handleUpload(files) {
        const formData = new FormData();
        for (const file of files) {
            formData.append('file', file);
        }

        document.body.classList.add('loading');
        uploadStatus.textContent = 'Uploading...';

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Upload failed');
            
            const result = await response.json();
            uploadStatus.textContent = 'Upload successful!';
            setTimeout(() => uploadStatus.textContent = '', 3000);
        } catch (error) {
            uploadStatus.textContent = `Upload failed: ${error.message}`;
        } finally {
            document.body.classList.remove('loading');
        }
    }

    // Search Functionality
    async function performSearch() {
        const query = searchInput.value.trim();
        if (!query) return;

        document.body.classList.add('loading');
        searchResults.innerHTML = '<div class="loading">Searching...</div>';

        try {
            const response = await fetch('/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });

            if (!response.ok) throw new Error('Search failed');
            
            const results = await response.json();
            displaySearchResults(results);
        } catch (error) {
            searchResults.innerHTML = `<div class="error">Search failed: ${error.message}</div>`;
        } finally {
            document.body.classList.remove('loading');
        }
    }

    function displaySearchResults(data) {
        searchResults.innerHTML = '';
        
        if (!data || !data.results || !Array.isArray(data.results) || data.results.length === 0) {
            searchResults.innerHTML = '<div class="no-results">No results found</div>';
            return;
        }
        
        data.results.forEach(result => {
            const resultDiv = document.createElement('div');
            resultDiv.className = 'search-result-item';
            
            const content = document.createElement('div');
            content.className = 'result-content';
            content.innerHTML = `
                <h4>${result.title || 'Untitled'}</h4>
                <p>${result.content || result.text || 'No content available'}</p>
                ${result.score ? `<div class="score">Relevance: ${result.score.toFixed(2)}</div>` : ''}
            `;
            
            resultDiv.appendChild(content);
            searchResults.appendChild(resultDiv);
        });
    }

    // Graph Visualization
    async function updateGraph() {
        document.body.classList.add('loading');
        const container = document.getElementById('graph');
        container.innerHTML = '<div class="loading">Loading graph...</div>';

        try {
            const response = await fetch('/graph');
            if (!response.ok) throw new Error('Failed to fetch graph data');
            
            const data = await response.json();
            if (!data || !data.nodes || !data.edges || data.nodes.length === 0) {
                container.innerHTML = '<div class="error">No graph data available. Try uploading some documents first.</div>';
                return;
            }
            
            console.log('Graph data:', data);  // Debug log
            visualizeGraph(data);
        } catch (error) {
            console.error('Error updating graph:', error);
            container.innerHTML = `<div class="error">Error creating graph: ${error.message}</div>`;
        } finally {
            document.body.classList.remove('loading');
        }
    }

    function visualizeGraph(data) {
        const container = document.getElementById('graph');
        
        // Add graph controls
        const controlsDiv = document.createElement('div');
        controlsDiv.className = 'graph-controls';
        controlsDiv.innerHTML = `
            <button class="action-button" id="zoomInBtn" title="Zoom In">
                <i class="fas fa-search-plus"></i>
            </button>
            <button class="action-button" id="zoomOutBtn" title="Zoom Out">
                <i class="fas fa-search-minus"></i>
            </button>
            <button class="action-button" id="fitBtn" title="Fit to View">
                <i class="fas fa-expand"></i>
            </button>
            <button class="action-button" id="centerBtn" title="Center Graph">
                <i class="fas fa-bullseye"></i>
            </button>
        `;
        container.parentNode.insertBefore(controlsDiv, container);
        
        // Create the data object for vis.js
        const graphData = {
            nodes: new vis.DataSet(data.nodes),
            edges: new vis.DataSet(data.edges)
        };
        
        const options = {
            nodes: {
                shape: 'dot',
                scaling: {
                    min: 10,
                    max: 30,
                    label: {
                        enabled: true,
                        min: 14,
                        max: 30,
                        maxVisible: 30,
                        drawThreshold: 5
                    }
                },
                font: {
                    size: 14,
                    face: 'Arial',
                    color: '#e6e6e6'
                },
                borderWidth: 2,
                shadow: true,
                color: {
                    background: '#00b3b3',
                    border: '#008080',
                    highlight: {
                        background: '#00ffff',
                        border: '#00b3b3'
                    },
                    hover: {
                        background: '#00e6e6',
                        border: '#00b3b3'
                    }
                }
            },
            edges: {
                width: 2,
                color: {
                    color: '#404040',
                    highlight: '#606060',
                    hover: '#505050'
                },
                smooth: {
                    type: 'continuous',
                    roundness: 0.5
                },
                scaling: {
                    min: 1,
                    max: 10,
                    label: {
                        enabled: true
                    }
                }
            },
            physics: {
                stabilization: {
                    iterations: 200,
                    fit: true
                },
                barnesHut: {
                    gravitationalConstant: -10000,
                    centralGravity: 0.3,
                    springLength: 200,
                    springConstant: 0.04,
                    damping: 0.09
                }
            },
            interaction: {
                hover: true,
                tooltipDelay: 300,
                zoomView: true,
                dragView: true,
                hideEdgesOnDrag: true,
                hideEdgesOnZoom: true,
                keyboard: {
                    enabled: true,
                    speed: {
                        x: 10,
                        y: 10,
                        zoom: 0.02
                    },
                    bindToWindow: false
                },
                navigationButtons: true,
                zoomSpeed: 1
            }
        };

        // Create the network
        try {
            const network = new vis.Network(container, graphData, options);
            
            // Add control button handlers
            document.getElementById('zoomInBtn').onclick = () => network.zoom(1.2);
            document.getElementById('zoomOutBtn').onclick = () => network.zoom(0.8);
            document.getElementById('fitBtn').onclick = () => network.fit();
            document.getElementById('centerBtn').onclick = () => network.moveTo({
                position: { x: 0, y: 0 },
                scale: 1.0
            });
            
            // Add keyboard shortcuts
            document.addEventListener('keydown', (event) => {
                if (event.target.tagName === 'INPUT') return; // Don't handle if typing in input
                
                switch(event.key) {
                    case '=':
                    case '+':
                        network.zoom(1.2);
                        break;
                    case '-':
                        network.zoom(0.8);
                        break;
                    case 'f':
                        network.fit();
                        break;
                    case 'c':
                        network.moveTo({ position: { x: 0, y: 0 }, scale: 1.0 });
                        break;
                }
            });
            
            console.log('Graph visualization created successfully');
        } catch (error) {
            console.error('Error creating graph:', error);
            container.innerHTML = 
                `<div class="error">Error creating graph: ${error.message}</div>`;
        }
    }

    // Event Listeners
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', e => {
        if (e.key === 'Enter') sendMessage();
    });

    fileInput.addEventListener('change', e => handleUpload(e.target.files));
    
    searchButton.addEventListener('click', performSearch);
    searchInput.addEventListener('keypress', e => {
        if (e.key === 'Enter') performSearch();
    });

    updateGraphBtn.addEventListener('click', updateGraph);

    // Initial graph load
    updateGraph();
}); 