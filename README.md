# FYP-StockForecasting

This project utilizes 3 different approaches to attempt tackling the problem of predicting stock market prices fluctations.

### Algorithmic Trading
For the first approach, I created functions performing traditional techniques used in technical analysis such as:
- [Fischer Transform](https://www.investopedia.com/terms/f/fisher-transform.asp)
- [Relative Strength Index](https://www.investopedia.com/terms/r/rsi.asp)
- [Simple Moving Averages](https://www.investopedia.com/terms/s/sma.asp)

I then used a combination of these techniques to deliver predictions on the price movements.


### Machine Learning
I then attempted to solve the problem through a Machine Learning approch. I created an algorithm that utilizes a [Neural Network](https://en.wikipedia.org/wiki/Artificial_neural_network) to predict stock market movements. 
The algorithm was trained using the daily closing prices of the index.


### Hybrid Approach
Finally, I combined both techniques into a Hybrid algorithm, that feeds the technical indicators into the Neural Network to improve the accuracy of the predictions.

*Note: Please keep in mind that when this code was written during my undergraduate years, I was still oblivious to SOLID principles as well as the use of proper source control*
