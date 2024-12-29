const express = require("express");
const session = require('express-session');
const pageRouter = require("./routes/pages");
const apiRouter = require("./routes/api");
const MongoStore = require('connect-mongo');

const path = require('path');


// Initialize app

const app = express();





app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use(
  session({
    secret: 'Secret',
    resave: false,
    saveUninitialized: false,
    store: MongoStore.create({ mongoUrl: 'your_mongofb_credentials' }),
  })
)

app.use(express.static(path.join(__dirname, 'public')));


app.use('/', pageRouter);
app.use('/api', apiRouter);


const PORT = process.env.PORT || 5001;
app.listen(PORT, () => {
  console.log(`Server started on port ${PORT}`);
})
