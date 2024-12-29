import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Box, Typography, Button } from '@mui/material';

function ApiTest() {
    const [pingStatus, setPingStatus] = useState('Not tested');
    const [chatStatus, setChatStatus] = useState('Not tested');

    const testPing = async () => {
        try {
            const response = await axios.get('http://localhost:8000/ping');
            setPingStatus(`Success: ${JSON.stringify(response.data)}`);
        } catch (error) {
            setPingStatus(`Error: ${error.message}`);
            console.error('Ping error:', error);
        }
    };

    const testChat = async () => {
        try {
            const response = await axios.post('http://localhost:8000/chat', {
                message: 'Test message'
            }, {
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            setChatStatus(`Success: ${JSON.stringify(response.data)}`);
        } catch (error) {
            setChatStatus(`Error: ${error.message}`);
            console.error('Chat error:', error);
        }
    };

    return (
        <Box sx={{ p: 2, bgcolor: '#f0f0f0', borderRadius: 2, mb: 2 }}>
            <Typography variant="h6">API Test</Typography>
            
            <Box sx={{ mt: 1 }}>
                <Button variant="contained" onClick={testPing} sx={{ mr: 1 }}>
                    Test Ping
                </Button>
                <Typography>Ping Status: {pingStatus}</Typography>
            </Box>
            
            <Box sx={{ mt: 2 }}>
                <Button variant="contained" onClick={testChat} sx={{ mr: 1 }}>
                    Test Chat
                </Button>
                <Typography>Chat Status: {chatStatus}</Typography>
            </Box>
        </Box>
    );
}

export default ApiTest; 