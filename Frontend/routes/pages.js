const express = require("express");
const router = express.Router();
const { isAuthenticated } = require('../middleware/auth');

router.get('/', (req, res) => res.sendFile('index.html', { root: 'public' }));
router.get('/login', (req, res) => res.sendFile('login.html', { root: 'public' }));
router.get('/register', (req, res) => res.sendFile('register.html', { root: 'public' }));
router.get('/user', isAuthenticated, (req, res) => res.sendFile('Real-time.html', { root: 'public' }));

module.exports = router;

