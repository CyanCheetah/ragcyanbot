import React, { useState, useRef, useEffect } from 'react';
import { 
  Container, 
  Paper, 
  TextField, 
  Button, 
  Box, 
  Typography,
  CircularProgress,
  ThemeProvider,
  createTheme
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import axios from 'axios';
import ApiTest from './ApiTest';

const theme = createTheme({
  palette: {
    primary: {
      main: '#2563eb',
    },
  },
});

function App() {
  const [messages, setMessages] = useState([{
    text: "Hello! I'm your RAG-powered assistant. How can I help you today?",
    isSystem: true
  }]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { text: userMessage, isUser: true }]);
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/chat', {
        message: userMessage
      }, {
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: 300000 // 5 minute timeout
      });

      if (response.data.error) {
        throw new Error(response.data.error);
      }

      setMessages(prev => [...prev, { 
        text: response.data.response, 
        isUser: false 
      }]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = error.response?.data?.error || 
                          error.message || 
                          'Failed to get response from server';
      setMessages(prev => [...prev, { 
        text: `Error: ${errorMessage}`, 
        isUser: false,
        isError: true 
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <Container maxWidth="md" sx={{ height: '100vh', py: 4 }}>
        <ApiTest />
        <Paper 
          elevation={3} 
          sx={{ 
            height: 'calc(100vh - 180px)',
            display: 'flex', 
            flexDirection: 'column',
            overflow: 'hidden',
            bgcolor: '#ffffff'
          }}
        >
          <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
            <Typography variant="h5" align="center" sx={{ fontWeight: 600 }}>
              RAG Chatbot
            </Typography>
            <Typography variant="subtitle1" align="center" color="text.secondary">
              Powered by LLaMA
            </Typography>
          </Box>

          <Box 
            sx={{ 
              flex: 1, 
              overflow: 'auto', 
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              gap: 1,
              bgcolor: '#f8fafc'
            }}
          >
            {messages.map((message, index) => (
              <Paper
                key={index}
                elevation={0}
                sx={{
                  p: 2,
                  maxWidth: '80%',
                  alignSelf: message.isUser ? 'flex-end' : 'flex-start',
                  bgcolor: message.isUser 
                    ? 'primary.main' 
                    : message.isSystem 
                      ? '#f1f5f9'
                      : message.isError 
                        ? '#fee2e2' 
                        : '#ffffff',
                  color: message.isUser ? '#ffffff' : '#1e293b',
                  borderRadius: 2,
                  boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
                }}
              >
                <Typography 
                  sx={{ 
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word'
                  }}
                >
                  {message.text}
                </Typography>
              </Paper>
            ))}
            <div ref={messagesEndRef} />
          </Box>

          <Box 
            component="form" 
            onSubmit={handleSubmit}
            sx={{ 
              p: 2, 
              borderTop: 1, 
              borderColor: 'divider',
              bgcolor: '#ffffff'
            }}
          >
            <Box sx={{ display: 'flex', gap: 1 }}>
              <TextField
                fullWidth
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type your message..."
                disabled={loading}
                variant="outlined"
                size="small"
                sx={{
                  '& .MuiOutlinedInput-root': {
                    borderRadius: 2,
                  }
                }}
              />
              <Button 
                type="submit" 
                variant="contained" 
                disabled={loading}
                sx={{
                  borderRadius: 2,
                  minWidth: '100px'
                }}
                endIcon={loading ? <CircularProgress size={20} color="inherit" /> : <SendIcon />}
              >
                Send
              </Button>
            </Box>
          </Box>
        </Paper>
      </Container>
    </ThemeProvider>
  );
}

export default App;
