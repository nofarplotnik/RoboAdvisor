import psycopg2
import psycopg2.extras
import http.client
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update, bot
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    CallbackContext,
)

from pandas_datareader import data as pdr
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

start_year = '2005-01-01'
end_year = '2022-06-22'
Num_porSimulation = 1000
V = 0

STOCK1,STOCK2,STOCK3,STOCK4,SELECTEDDEF,GORM,MARKOWITZ,GINI=range(8)

def start(update: Update, context: CallbackContext) -> int:
    reply_keyboard = [['מעולה', 'גרוע']]
    update.message.reply_text(  'שים לב להתחלה מחדש לחץ /start '
                                '\n'
                                'לסיום לחץ /cancel ' ''
                                ' \n')
    update.message.reply_text(  'האפליקציה פותחה כחלק מקורס פינטק באקדמית תל אביב-יפו במסגרת אקדמאית לימודית. אין לראות בהמלצות המלצות אמיתיות, אלא יש להתייעץ עם יועץ מוסמך.'
                                )
    update.message.reply_text(
        'היי אני Robo Advisor, נעים מאוד להכיר ! \n'
        'אני כאן בכדי לעזור להתאים שלושה מסלולי השקעות עבורך. \n מה שלומך היום ? '
        '  ',
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True, input_field_placeholder='מעולה או גרוע?'
        ),
    )
    return STOCK1

def stock1(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    if update.message.text == "מעולה":
        update.message.reply_text(
            'תענוג לשמוע, יאללה ממשיכים!',
        )
    elif update.message.text == "גרוע":
        update.message.reply_text(
            'מצטער לשמוע, החוויה בבוט בטוח תשפר את מצב הרוח!',
        )
    update.message.reply_text(
        "שים לב יש להכניס שמות מדויקים של ארבע מניות אחרת לא אצליח להתאים את המסלולים",
        reply_markup=ReplyKeyboardRemove(),

    )
    update.message.reply_text(
         "הכנס מניה מספר 1",
         reply_markup=ReplyKeyboardRemove(),

    )
    return STOCK2

def stock2(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    global selected
    selected=[]
    selected.append(update.message.text)
    update.message.reply_text(
        "הכנס מניה מספר 2",
        reply_markup=ReplyKeyboardRemove(),
    )
    return STOCK3

def stock3(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    selected.append(update.message.text)
    update.message.reply_text(
        "הכנס מניה מספר 3",
        reply_markup=ReplyKeyboardRemove(),
    )
    return STOCK4

def stock4(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    selected.append(update.message.text)
    update.message.reply_text(
        "הכנס מניה מספר 4",
        reply_markup=ReplyKeyboardRemove(),
    )
    return SELECTEDDEF
def selecteddef(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    selected.append(update.message.text)
    update.message.reply_text(
        "הקלד 1 כדי להמשיך",
        reply_markup=ReplyKeyboardRemove(),
    )
    return GORM
def gorm(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    reply_keyboard = [['1', '2']]
    update.message.reply_text(
         "לחץ 1 עבור גיני, 2 עבור מרקוביץ",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True, input_field_placeholder='1 עבור גיני 2 עבור מרקוביץ'
        ),
    )
    return MARKOWITZ

def markowitz(update: Update, context: CallbackContext) -> int:
    if update.message.text == "2":
        yf.pdr_override()
        frame = {}
        for stock in selected:
                data_var = pdr.get_data_yahoo(stock, start_year, end_year)['Adj Close']
                data_var.to_frame()
                frame.update({stock: data_var})

        table = pd.DataFrame(frame)
        returns_daily = table.pct_change()
        returns_annual = ((1 + returns_daily.mean()) ** 250) - 1

        cov_daily = returns_daily.cov()
        cov_annual = cov_daily * 250
        port_returns = []
        port_volatility = []
        sharpe_ratio = []
        stock_weights = []
        num_assets = len(selected)
        num_portfolios = Num_porSimulation
        np.random.seed(101)
        for single_portfolio in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            returns = np.dot(weights, returns_annual)
            returns = np.dot(weights, returns_annual)
            volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
            sharpe = returns / volatility
            sharpe_ratio.append(sharpe)
            port_returns.append(returns * 100)
            port_volatility.append(volatility * 100)
            stock_weights.append(weights)

        portfolio = {'Returns': port_returns,
                     'Volatility': port_volatility,
                     'Sharpe Ratio': sharpe_ratio}

        for counter, symbol in enumerate(selected):
            portfolio[symbol + ' Weight'] = [Weight[counter] for Weight in stock_weights]

        df = pd.DataFrame(portfolio)

        column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock + ' Weight' for stock in selected]

        df = df[column_order]

        min_volatility = df['Volatility'].min()
        max_sharpe = df['Sharpe Ratio'].max()
        max_return = df['Returns'].max()
        max_vol = df['Volatility'].max()
        sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
        min_variance_port = df.loc[df['Volatility'] == min_volatility]
        max_returns = df.loc[df['Returns'] == max_return]
        max_vols = df.loc[df['Volatility'] == max_vol]

        red_num = df.index[df["Returns"] == max_return]
        yellow_num = df.index[df['Volatility'] == min_volatility]
        green_num = df.index[df['Sharpe Ratio'] == max_sharpe]
        multseries = pd.Series([1, 1, 1] + [100 for stock in selected],
                               index=['Returns', 'Volatility', 'Sharpe Ratio'] + [stock + ' Weight' for stock in selected])
        with pd.option_context('display.float_format', '%{:,.2f}'.format):
            Max_returns_Porfolio=df.loc[red_num[0]].multiply(multseries).to_string()
            Safest_Portfolio=df.loc[yellow_num[0]].multiply(multseries).to_string()
            Sharpe_Portfolio=df.loc[green_num[0]].multiply(multseries).to_string()
        update.message.reply_text(

            "Max returns Porfolio:" + '\n' + Max_returns_Porfolio+  '\n'+  '\n'+ "Safest Portfolio:" +  '\n'+ Safest_Portfolio + '\n'+ '\n'+ "Sharpe Portfolio:" + '\n' + Sharpe_Portfolio
        , reply_markup=ReplyKeyboardRemove()
        )
        update.message.reply_text(
            'האפליקציה פותחה כחלק מקורס פינטק באקדמית תל אביב-יפו במסגרת אקדמאית לימודית. אין לראות בהמלצות המלצות אמיתיות, אלא יש להתייעץ עם יועץ מוסמך.'
            )
        update.message.reply_text(
            'השקעה בטוחה להתראות!', reply_markup=ReplyKeyboardRemove()
        )
        return ConversationHandler.END
    if update.message.text == "1":
        update.message.reply_text('אנא רשום את הרגישות שלך לסיכון. כאשר סיכון נמוך הכוונה שכמעט אינך רגיש לסיכון אלא מסתכל בעיקר על התשואות. וכאשר תגדיר סיכון גבוה הפירוש שהוא רגיש מאוד לתנודתיות השוק.'
        '\n'
        'לחץ 1 לסיכון נמוך, 2 לסיכון בינוני או 3 לסיכון גבוה.'
        ,
            reply_markup=ReplyKeyboardRemove()
        )
        return GINI
    return ConversationHandler.END

def gini(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    if update.message.text=="1":
        V=1.25
    elif update.message.text=="2":
        V = 2
    elif update.message.text == "3":
        V = 4
    yf.pdr_override()
    frame = {}
    for stock in selected:
        data_var = pdr.get_data_yahoo(stock, start_year, end_year)['Adj Close']
        data_var.to_frame()
        frame.update({stock: data_var})

    # Mathematical calculations, creation of 5000 portfolios,
    table = pd.DataFrame(frame)
    returns_daily = table.pct_change()

    port_profolio_annual = []
    port_gini_annual = []
    sharpe_ratio = []
    stock_weights = []

    # set the number of combinations for imaginary portfolios
    num_assets = len(selected)
    num_portfolios = Num_porSimulation

    # set random seed for reproduction's sake
    np.random.seed(101)

    # Mathematical calculations, creation of 5000 portfolios,
    table = pd.DataFrame(frame)
    returns_daily = table.pct_change()
    for stock in returns_daily.keys():
        table[stock + '_change'] = returns_daily[stock]

    # populate the empty lists with each portfolios returns,risk and weights
    for single_portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        profolio = np.dot(returns_daily, weights)
        profolio_return = pd.DataFrame(profolio)
        rank = profolio_return.rank()
        rank_divided_N = rank / len(rank)  # Rank/N
        one_sub_rank_divided_N = 1 - rank_divided_N  # 1-Rank/N
        one_sub_rank_divided_N_power_v_sub_one = one_sub_rank_divided_N ** (V - 1)  # (1-Rank/N)^(V-1)
        mue = profolio_return.mean().tolist()[0]
        x_avg = one_sub_rank_divided_N_power_v_sub_one.mean().tolist()[0]
        profolio_mue = profolio_return - mue
        rank_sub_x_avg = one_sub_rank_divided_N_power_v_sub_one - x_avg
        profolio_mue_mult_rank_x_avg = profolio_mue * rank_sub_x_avg
        summary = profolio_mue_mult_rank_x_avg.sum().tolist()[0] / (len(rank) - 1)
        gini_daily = summary * (-V)
        gini_annual = gini_daily * (254 ** 0.5)
        profolio_annual = ((1 + mue) ** 254) - 1
        sharpe = profolio_annual / gini_annual
        sharpe_ratio.append(sharpe)
        port_profolio_annual.append(profolio_annual * 100)
        port_gini_annual.append(gini_annual * 100)
        stock_weights.append(weights)
    # a dictionary for Returns and Risk values of each portfolio
    portfolio = {'Profolio_annual': port_profolio_annual,
                 'Gini': port_gini_annual,
                 'Sharpe Ratio': sharpe_ratio}

    # extend original dictionary to accomodate each ticker and weight in the portfolio
    for counter, symbol in enumerate(selected):
        portfolio[symbol + ' Weight'] = [Weight[counter] for Weight in stock_weights]

    # make a nice dataframe of the extended dictionary
    df = pd.DataFrame(portfolio)

    # get better labels for desired arrangement of columns
    column_order = ['Profolio_annual', 'Gini', 'Sharpe Ratio'] + [stock + ' Weight' for stock in selected]

    # reorder dataframe columns
    df = df[column_order]

    # plot frontier, max sharpe & min Gini values with a scatterplot
    # find min Gini & max sharpe values in the dataframe (df)
    min_gini = df['Gini'].min()
    max_sharpe = df['Sharpe Ratio'].max()
    max_profolio_annual = df['Profolio_annual'].max()
    max_gini = df['Gini'].max()

    # use the min, max values to locate and create the two special portfolios
    sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
    min_variance_port = df.loc[df['Gini'] == min_gini]
    max_profolios_annual = df.loc[df['Profolio_annual'] == max_profolio_annual]
    max_ginis = df.loc[df['Gini'] == max_gini]

    red_num = df.index[df["Profolio_annual"] == max_profolio_annual]
    yellow_num = df.index[df['Gini'] == min_gini]
    green_num = df.index[df['Sharpe Ratio'] == max_sharpe]
    multseries = pd.Series([1, 1, 1] + [100 for stock in selected],
                           index=['Profolio_annual', 'Gini', 'Sharpe Ratio'] + [stock + ' Weight' for stock in
                                                                                selected])
    with pd.option_context('display.float_format', '%{:,.2f}'.format):
        Max_returns_Porfolio = df.loc[red_num[0]].multiply(multseries).to_string()
        Safest_Portfolio = df.loc[yellow_num[0]].multiply(multseries).to_string()
        Sharpe_Portfolio = df.loc[green_num[0]].multiply(multseries).to_string()
    update.message.reply_text(

        "Max returns Porfolio:" + '\n' + Max_returns_Porfolio + '\n' + '\n' + "Safest Portfolio:" + '\n' + Safest_Portfolio + '\n' + '\n' + "Sharpe Portfolio:" + '\n' + Sharpe_Portfolio
        , reply_markup=ReplyKeyboardRemove()
    )
    update.message.reply_text(  'האפליקציה פותחה כחלק מקורס פינטק באקדמית תל אביב-יפו במסגרת אקדמאית לימודית. אין לראות בהמלצות המלצות אמיתיות, אלא יש להתייעץ עם יועץ מוסמך.'
                                )
    update.message.reply_text(
        'השקעה בטוחה להתראות!', reply_markup=ReplyKeyboardRemove()
    )

    return ConversationHandler.END




def cancel(update: Update, context: CallbackContext) -> int:
    """Cancels and ends the conversation."""
    user = update.message.from_user
    update.message.reply_text(  'האפליקציה פותחה כחלק מקורס פינטק באקדמית תל אביב-יפו במסגרת אקדמאית לימודית. אין לראות בהמלצות המלצות אמיתיות, אלא יש להתייעץ עם יועץ מוסמך.'
                                )
    update.message.reply_text(
        'השקעה בטוחה להתראות!', reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END
def main() -> None:
    updater = Updater("5482784911:AAE5hqW7SUDmwjKX_5Hb_XKy8tpIP4eUp8k")
    dispatcher = updater.dispatcher
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            STOCK1: [MessageHandler(Filters.regex('^(מעולה|גרוע)$'), stock1)],
            STOCK2: [MessageHandler(Filters.text & ~Filters.command, stock2)],
            STOCK3: [MessageHandler(Filters.text & ~Filters.command, stock3)],
            STOCK4: [MessageHandler(Filters.text & ~Filters.command, stock4)],
            SELECTEDDEF: [MessageHandler(Filters.text & ~Filters.command, selecteddef)],
            GORM: [MessageHandler(Filters.regex('^(1|2)$'), gorm)],
            MARKOWITZ: [MessageHandler(Filters.text & ~Filters.command, markowitz)],
            GINI: [MessageHandler(Filters.text & ~Filters.command, gini)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )
    dispatcher.add_handler(conv_handler)
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
