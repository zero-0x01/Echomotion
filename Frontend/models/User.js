const mongoose = require('mongoose');

const accountSchema = new mongoose.Schema({
    username: { type: String, required: true, unique: true },
    password: { type: String, required: true },
}, { collection: 'accounts' });

const User = mongoose.model('User', userSchema);

module.exports = User;