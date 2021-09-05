# stock-support-alerts
A trading bot that gives SMS alerts to buy when a stock crosses reaches a support line.

This strategy is predicated around the fact that stocks often tend to rebound off strong support lines/prices that have been established over years. For example, take a look at INTEL's stock price below and how it has respected a strong two-year support line. This trading bot would be able to notify you when INTEL, amongst thousands of other stocks, reach a support line -- thus, giving you a signal to buy.

<img width="1349" alt="Screenshot 2021-09-05 at 20 51 04" src="https://user-images.githubusercontent.com/79874741/132134863-f6d526b3-2caf-4c31-ab4e-ce0912b2c1bc.png">

For this to work, I recommend running script.py on repl.it so that you can run the script perpetually without fear of it crashing. Unfortunately, repl.it automatically stops scripts from running if it they are inactive for over an hour or so. Since this script only runs once a day, it falls into that category of so-called inactive scripts. To circumvent this problem, you need the webserver.py file as well. Furthermore, you need to link the script to Uptimerobot so that it doesn't appear to be inactive from repl.it's side.
