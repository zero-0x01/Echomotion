const mongoose = require('mongoose');

// Connect to MongoDB
mongoose.connect('Your_mongodb', {
    useNewUrlParser: true,
    useUnifiedTopology: true,
})
    .then(() => {
        console.log('Connected to MongoDB');
    })
    .catch((err) => {
        console.error('Failed to connect to MongoDB:', err);
    });

// Define the schema for the "accounts" collection
const accountSchema = new mongoose.Schema({
    username: { type: String, required: true, unique: true },
    password: { type: String, required: true },
}, { collection: 'accounts' });

// Create the model for the "accounts" collection
const Account = mongoose.model('Account', accountSchema);

// Function to find a user by username
async function findByUsername(username) {
    try {
        // Find the user by username in the 'accounts' collection
        const user = await Account.findOne({ username });
        return user || null;
    } catch (err) {
        console.error('Error finding user:', err);
        return null;
    }
}

module.exports = { findByUsername, Account };
