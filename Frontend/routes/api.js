const express = require('express');
const router = express.Router();
const multer = require('multer');
const axios = require('axios');
const fs = require('fs');
const path = require('path');  // Use path module for handling file paths
const FormData = require('form-data');  // Use form-data module to send multipart requests







// Multer setup for file uploads (store files in /tmp directory on Heroku)
const upload = multer({ dest: '/tmp/' }); // Use Heroku's tmp directory

// Route to handle voice prediction API
router.post('/predict-emotion', upload.single('audio'), async (req, res) => {
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
