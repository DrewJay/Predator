var path = require('path');

module.exports = {
  devServer: {
    contentBase: path.join(__dirname, '/'),
    compress: true,
    port: 8080
  }
};