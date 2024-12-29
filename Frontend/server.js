const express = require("express");
const pageRouter = require("./routes/pages");
const apiRouter = require("./routes/api");

require('dotenv').config()
const path = require('path');


// Initialize app

const app = express();




app.use(express.json());
app.use(express.urlencoded({ extended: true }));



app.use(express.static(path.join(__dirname, 'public')));


app.use('/', pageRouter);
app.use('/api', apiRouter);


const PORT = process.env.PORT || 5001;
app.listen(PORT, () => {
  console.log(`Server started on port ${PORT}`);
})
