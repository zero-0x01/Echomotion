const express = require('express');
const router = express.Router();
const multer = require('multer');
const axios = require('axios');
const fs = require('fs');
const path = require('path');  // Use path module for handling file paths
const FormData = require('form-data');  // Use form-data module to send multipart requests
const { isAuthenticated } = require('../middleware/auth');
const { findByUsername, Account } = require('../database/db');


// Register Route
router.post('/register', async (req, res) => {
  const { username, password } = req.body;

  if (!username || !password) {
    return res.status(400).json({ message: 'Username and password are required' });
  }

  try {
    // Check if the user already exists
    const existingUser = await findByUsername(username);
    if (existingUser) {
      return res.status(409).json({ message: 'Username already exists' });
    }

    // Create a new user
    const newUser = new Account({ username, password });
    await newUser.save();

    res.status(201).json({ message: 'User registered successfully' });
  } catch (err) {
    console.error('Error during registration:', err);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Login route (example authentication route)
router.post('/login', async (req, res) => {
  const { username, password } = req.body;

  try {
    const user = await findByUsername(username);
    if (!user || user.password !== password) {
      return res.status(401).json({ message: 'Invalid username or password' });
    }

    req.session.user = { id: user._id, username: user.username };
    res.status(200).json({ message: 'Login successful' });
  } catch (err) {
    console.error('Error during login:', err);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Logout route
router.post('/logout', (req, res) => {
  req.session.destroy(() => {
    res.json({ message: "Logged out successfully" });
  })
});

// Multer setup for file uploads (store files in /tmp directory on Heroku)
const upload = multer({ dest: '/tmp/' }); // Use Heroku's tmp directory

// Route to handle voice prediction API
router.post('/predict-emotion', isAuthenticated, upload.single('audio'), async (req, res) => {
  try {
    // Ensure the file is uploaded
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    // File path in Heroku's /tmp directory
    const audioFilePath = path.join('/tmp', req.file.filename);

    // Form data to send the file to Flask server for processing
    const formData = new FormData();
    formData.append('audio', fs.createReadStream(audioFilePath));

    // Send the audio file to Flask API for emotion prediction
    const response = await axios.post('http://127.0.0.1:5000/predict-emotion', formData, {
      headers: formData.getHeaders(),  // Use formData.getHeaders() to get the correct content type
    });

    // Clean up the uploaded file after prediction
    fs.unlinkSync(audioFilePath); // Delete the temporary file from /tmp

    // Return the predicted emotion from the Flask response
    res.json({ predicted_emotion: response.data.predicted_emotion });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'An error occurred while processing the audio file.' });
  }
});

module.exports = router;
