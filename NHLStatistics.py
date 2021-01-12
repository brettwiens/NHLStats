import streamlit as st
import pandas as pd
import numpy as np
# from urllib2 import Request, urlopen
import requests
#import itertools
#import json
#from itertools import chain
from io import StringIO
#import ast
import matplotlib
import matplotlib.pyplot as plt
color_map = plt.cm.winter
#from scipy.stats import kde
from matplotlib.patches import RegularPolygon
#import math
from PIL import Image
import seaborn as sns
#import matplotlib.image as mpimg
#from matplotlib.pyplot import show
from tqdm.notebook import tqdm, trange
from keras.models import load_model

# from tqdm.notebook import tqdm, trange
#import time
st.set_page_config(layout='wide')

#from pandas.io.json import json_normalize


# Options
year = '2019'
season_type = '02' # Regular Season = 02 / Playoffs = 03 / Pre-Season = 01
# max_game_ID = 1290 ## 1290 Games/Season

# GoalieStats = pd.read_hdf("GoalieStats.h5", "GoalieStatistics")
# SkaterStats = pd.read_hdf("SkaterStats.h5", "SkaterStatistics")
# Goals = pd.read_hdf("Goals.h5", "Goals")
# Shots = pd.read_hdf("Shots.h5", "Shots")
# Hits = pd.read_hdf("Hits.h5", "Hits")
# Missed = pd.read_hdf("Missed.h5", "Missed_Shots")
# Penalties = pd.read_hdf("Penalties.h5", "Penalties")
# NHLRoster = pd.read_hdf("NHLRoster.h5", "FullRoster")
# PlayerDetails = pd.read_hdf("PlayerDetails.h5", "PlayerDetails")

GoalieStats = pd.read_csv("GoalieStats.csv")
GoalieStats = GoalieStats.drop(['Unnamed: 0'], axis = 1)
# SkaterStats = pd.read_csv("SkaterStats.csv")
# SkaterStats = SkaterStats.drop(['Unnamed: 0'], axis = 1)
Goals = pd.read_csv("Goals.csv")
Goals = Goals.drop(['Unnamed: 0'], axis = 1)
Shots = pd.read_csv("Shots.csv")
Shots = Shots.drop(['Unnamed: 0'], axis = 1)
Hits = pd.read_csv("Hits.csv")
Hits = Hits.drop(['Unnamed: 0'], axis = 1)
Missed = pd.read_csv("Missed.csv")
Missed = Missed.drop(['Unnamed: 0'], axis = 1)
Penalties = pd.read_csv("Penalties.csv")
Penalties = Penalties.drop(['Unnamed: 0'], axis = 1)
NHLRoster = pd.read_csv("NHLRoster.csv")
NHLRoster = NHLRoster.drop(['Unnamed: 0'], axis = 1)
PlayerDetails = pd.read_csv("PlayerDetails.csv")
PlayerDetails = PlayerDetails.drop(['Unnamed: 0'], axis = 1)

def IceMaker(StatFrame):
    StatTable = StatFrame[['X', 'Y']]
    StatTable = StatTable.groupby(["X", "Y"]).size().reset_index(name="Freq")

    x = StatTable['X']
    y = StatTable['Y']
    for index, value in x.items():
        if x[index] < 0:
            x[index] = -x[index]
            y[index] = -y[index]

    freq = StatTable['Freq']
    sns.set()
    fig, ax = plt.subplots(frameon=True)

    DPI = 64
    IMG_WIDTH = 1000
    IMG_HEIGHT = 850
    fig = plt.figure(figsize=(IMG_WIDTH / DPI, IMG_HEIGHT / DPI), dpi=DPI)
    ax_extent = [0, 100, -42.5, 42.5]
    img = Image.open('HalfNHLArena.png')
    plt.imshow(img, extent=ax_extent)

    sns.set_style("white")
    ax = sns.kdeplot(x=x, y=y, cmap="icefire", fill=True, thresh=0.05, levels=100, zorder=2, alpha=0.5)
    sns.scatterplot(x=x, y=y, s=50, alpha=1, hue=freq, palette="dark:salmon_r")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(0.0)  # Remove the labelling of axes
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.grid: True
    ax.edgecolor: .8
    ax.linewidth: 1
    ax.set(xlim=(0, 100), ylim=(-42.5, 42.5))
    p = matplotlib.patches.Rectangle(xy=[0, -42.5], width=100, height=85, transform=ax.transData,
                                     facecolor="xkcd:white", alpha=0.3, zorder=-1)
    for col in ax.collections:
        col.set_clip_path(p)
    # plt.axis('off')
    return fig

TeamIndex = {
    'Anaheim Ducks': 'ANA.png',
    'Arizona Coyotes': 'ARI.png',
    'Boston Bruins': 'BOS.png',
    'Buffalo Sabres': 'BUF.png',
    'Calgary Flames': 'CGY.png',
    'Carolina Hurricanes': 'CAR.png',
    'Chicago Blackhawks': 'CHI.png',
    'Columbus Blue Jackets': 'CBJ.png',
    'Dallas Stars': 'DAL.PNG',
    'Detroit Red Wings': 'DET.png',
    'Edmonton Oilers': 'EDM.png',
    'Florida Panthers': 'FLA.png',
    'Los Angeles Kings': 'LAK.png',
    'Minnesota Wild': 'MIN.png',
    'Montréal Canadiens': 'MTL.png',
    'Nashville Predators': 'NSH.png',
    'New Jersey Devils': 'NJD.png',
    'New York Islanders': 'NYI.png',
    'New York Rangers': 'NYR.png',
    'Ottawa Senators': 'OTT.png',
    'Philadelphia Flyers': 'PHI.png',
    'Pittsburgh Penguins': 'PIT.png',
    'San Jose Sharks': 'SJS.png',
    'St. Louis Blues': 'STL.png',
    'Tampa Bay Lightning': 'TBL.png',
    'Toronto Maple Leafs': "'TOR.png'",
    'Vancouver Canucks': 'VAN.png',
    'Vegas Golden Knights': 'VGK.png',
    'Washington Capitals': 'WSH.png',
    'Winnipeg Jets': 'WPG.png',
}

##################
# User Interface #
##################

st.sidebar.image('BW_Spine_2.png', width=100)

Page = st.sidebar.radio("Select Page:", ['Standings', 'Player Statistics', 'Player Visualizations', 'Predictive Analytics'])

if(Page == 'Standings'):

    SeasonPick = st.sidebar.select_slider("Season:",
                                  ['19992000', '20002001', '20012002', '20022003', '20032004',
                                   '20052006', '20062007', '20072008', '20082009',
                                   '20092010', '20102011', '20112012', '20122013', '20132014',
                                   '20142015', '20152016', '20162017', '20172018', '20182019', '20192020', 'Current'],
                                  value='Current',)

    LeagueTable = pd.DataFrame()
    if SeasonPick == 'Current':
        try:
            StandingsData = pd.DataFrame.from_dict(requests.get('https://statsapi.web.nhl.com/api/v1/standings', verify=False).json())
            for Division in StandingsData['records']:
                #     print(Division)
                DivisionName = Division['division']['name']
                Conference = Division['conference']['name']
            st.header("Current Season")
        except:
            StandingsData = pd.DataFrame.from_dict(requests.get('https://statsapi.web.nhl.com/api/v1/standings?season=20192020', verify=False).json())
            st.header("Current Season Hasn't Begun - Showing 2019-2020")
    else:
        StandingsData = pd.DataFrame.from_dict(requests.get('https://statsapi.web.nhl.com/api/v1/standings?season=' + SeasonPick, verify=False).json())
        st.header(SeasonPick + " Season")
    for Division in StandingsData['records']:
        #     print(Division)
        DivisionName = Division['division']['name']
        Conference = Division['conference']['name']
        for Team in (Division['teamRecords']):
            CurrentTeam = StringIO("""Name;Division;Conference;Wins;Losses;Overtime;Goals For;Goals Against;Points;Division Rank;Conference Rank;League Rank;Games Played
            """ +
                                   str(Team['team']['name']) + ";" +
                                   str(Division['division']['name']) + ";" +
                                   str(Division['conference']['name']) + ";" +
                                   str(Team['leagueRecord']['wins']) + ";" +
                                   str(Team['leagueRecord']['losses']) + ";" +
                                   str(Team['leagueRecord']['ot']) + ";" +
                                   str(Team['goalsScored']) + ";" +
                                   str(Team['goalsAgainst']) + ";" +
                                   str(Team['points']) + ";" +
                                   str(Team['divisionRank']) + ";" +
                                   str(Team['conferenceRank']) + ";" +
                                   str(Team['leagueRank']) + ";" +
                                   str(Team['gamesPlayed']))
            CurrentTeam = pd.read_csv(CurrentTeam, sep=";")

            LeagueTable = LeagueTable.append(CurrentTeam, ignore_index=True)

    StatType = st.radio("Standings Type:", ['Division', 'Conference', 'League'])

    if StatType == 'League':
        st.subheader("League")
        st.write(LeagueTable.sort_values(by = ['League Rank']).style.background_gradient(cmap=sns.diverging_palette(220, 20, as_cmap=True)))

    if StatType == 'Conference':
        for Conference in LeagueTable.Conference.unique():
            st.subheader(Conference)
            st.write(LeagueTable.loc[LeagueTable['Conference'] == Conference].sort_values(by = ['Conference Rank']).style.background_gradient(cmap=sns.diverging_palette(220, 20, as_cmap=True)))

    if StatType == 'Division':
        for Division in LeagueTable.Division.unique():
            st.subheader(Division)
            st.write(LeagueTable.loc[LeagueTable['Division'] == Division].style.background_gradient(cmap=sns.diverging_palette(220, 20, as_cmap=True)))

if(Page == 'Player Statistics'):
    Position = st.sidebar.radio("Skaters or Goalies:", ['Skaters','Goalies'])
    if Position == 'Skaters':
        SeasonPick = st.sidebar.select_slider("Season:",
                                              [20052006, 20062007, 20072008, 20082009,
                                               20092010, 20102011, 20112012, 20122013, 20132014,
                                               20142015, 20152016, 20162017, 20172018, 20182019, 20192020,
                                               20202021],
                                              value=20192020, )
        SkaterStats = pd.read_csv("SkaterSeasonStats.csv")
        SkaterStats = SkaterStats.drop(['Unnamed: 0'], axis=1)
        SkaterStats = SkaterStats.loc[SkaterStats['Season'] == SeasonPick]
        st.subheader("Skater Statistics")
        st.dataframe(SkaterStats.sort_values(by = ['P'], ascending=False).style.background_gradient(cmap=sns.diverging_palette(220, 20, as_cmap=True)), height=600)
    else:
        SeasonPick = st.sidebar.select_slider("Season:",
                                              [20052006, 20062007, 20072008, 20082009,
                                               20092010, 20102011, 20112012, 20122013, 20132014,
                                               20142015, 20152016, 20162017, 20172018, 20182019, 20192020,
                                               20202021],
                                              value=20192020, )
        GoalieStats = pd.read_csv("GoalieSeasonStats.csv")
        GoalieStats = GoalieStats.drop(['Unnamed: 0'], axis=1)
        GoalieStats = GoalieStats.loc[GoalieStats['Season'] == SeasonPick]
        st.subheader("Goalie Statistics")
        st.dataframe(GoalieStats.sort_values(by = ['W'], ascending=False).style.background_gradient(cmap=sns.diverging_palette(220, 20, as_cmap=True)), height=600)

if(Page == 'Player Visualizations'):
    TeamSelect = st.sidebar.selectbox('Select a Team', np.sort(np.append(pd.unique(NHLRoster['Team']),"All")))
    PositionSelect = st.sidebar.selectbox('Select a Position', np.sort(np.append("All",pd.unique(NHLRoster['PosCode']))))

    if PositionSelect != "All":
        if TeamSelect != "All":
            PlayerSelect = st.sidebar.selectbox('Select a Player',
                                        NHLRoster.loc[(NHLRoster['PosCode'] == PositionSelect) &
                                                      (NHLRoster['Team'] == TeamSelect)]['FullName'].
                                        reset_index().
                                        drop(labels="index", axis=1), index=0)
        else:
            PlayerSelect = st.sidebar.selectbox('Select a Player',
                                        NHLRoster.loc[NHLRoster['PosCode'] == PositionSelect]['FullName'].sort_values().
                                        reset_index().
                                        drop(labels="index", axis=1), index=0)
    else:
        if TeamSelect != "All":
            PlayerSelect = st.sidebar.selectbox('Select a Player',
                                        NHLRoster.loc[NHLRoster['Team'] == TeamSelect]['FullName'].
                                        reset_index().
                                        drop(labels="index", axis=1), index=0)
        else:
            PlayerSelect = st.sidebar.selectbox('Select a Player',
                                        NHLRoster['FullName'].sort_values().
                                        reset_index().
                                        drop(labels="index", axis=1), index=0)

    SelectedPlayer = NHLRoster.loc[NHLRoster['FullName'] == PlayerSelect].reset_index()

    col1, col2 = st.beta_columns([2, 20])
    with col1:
        st.image('./TeamLogos/' + TeamIndex.get(NHLRoster.loc[NHLRoster['FullName'] == PlayerSelect]['Team'].to_string(index=False).lstrip()).strip('"\''), width=100)
    with col2:
        st.title(PlayerSelect)

    if SelectedPlayer['PosCode'][0] != "G":
        PlayerShotTotal = len(Shots.loc[Shots['P1Name'] == PlayerSelect])
        PlayerGoalTotal = len(Goals.loc[Goals['P1Name'] == PlayerSelect])
        if PlayerShotTotal == 0:
            PlayerShotPercentage = 0
        else:
            PlayerShotPercentage = PlayerGoalTotal / PlayerShotTotal
        LeagueShotTotal = len(Shots)
        LeagueShotPercentage = len(Goals) / len(Shots)

        st.sidebar.subheader("Quick Stats:")
        st.sidebar.write("Goals: " + str(PlayerGoalTotal))
        st.sidebar.write("League Goals: " + str(len(Goals)))
        st.sidebar.write("Shots: " + str(PlayerShotTotal))
        st.sidebar.write("League Shots: " + str(len(Shots)))
        st.sidebar.write("Shooting Percentage: " + str(round(PlayerShotPercentage, 3) * 100))
        st.sidebar.write("League Shooting Percentage: " + str(round(LeagueShotPercentage, 3) * 100))

        col1, col2, col3 = st.beta_columns(3)
        with col1:
            st.header("Goals")
            st.pyplot(IceMaker(Goals.loc[Goals['P1Name'] == PlayerSelect]), use_column_width=True)
        with col2:
            st.header("Shots")
            st.pyplot(IceMaker(Shots.loc[Shots['P1Name'] == PlayerSelect]), use_column_width=True)
        with col3:
            st.header("Hits")
            st.pyplot(IceMaker(Hits.loc[Hits['P1Name'] == PlayerSelect]), use_column_width=True)

        col1, col2, col3 = st.beta_columns(3)
        with col1:
            st.dataframe(Goals.loc[Goals['P1Name'] == PlayerSelect], height=500)
        with col2:
            st.dataframe(Shots.loc[Shots['P1Name'] == PlayerSelect], height=500)
        with col3:
            st.dataframe(Hits.loc[Hits['P1Name'] == PlayerSelect], height=500)

    else:
        GoalieSaves = len(Shots.loc[Shots['P2Name'] == PlayerSelect])
        GoalieGoalsAgainst = len(Goals.loc[Goals['P2Name'] == PlayerSelect].
                                 append(Goals.loc[Goals['P3Name'] == PlayerSelect]).
                                 append(Goals.loc[Goals['P4Name'] == PlayerSelect]))
        if GoalieSaves == 0:
            GoalieSavePercentage = 0
        else:
            GoalieSavePercentage = round(GoalieGoalsAgainst/GoalieSaves, 3)

        st.sidebar.subheader("Quick Stats:")
        st.sidebar.write("Saves: " + str(GoalieSaves))
        st.sidebar.write("Goals Against: " + str(GoalieGoalsAgainst))
        st.sidebar.write("Save Percentage: " + str((1-round(GoalieSavePercentage, 2))*100))

        col1, col2, col3 = st.beta_columns([5, 5, 5])
        with col1:
            st.header("Goals Against")
            # st.image('./NHLArena.png', use_column_width=True)
            st.pyplot(IceMaker(Goals.loc[Goals['P2Name'] == PlayerSelect].
                         append(Goals.loc[Goals['P3Name'] == PlayerSelect]).
                         append(Goals.loc[Goals['P4Name'] == PlayerSelect])), use_column_width=True)
        with col2:
            st.header("Shots Against")
            # st.image('./NHLArena.png', use_column_width=True)
            st.pyplot(IceMaker(Shots.loc[Shots['P2Name'] == PlayerSelect]), use_column_width=True)

        col1, col2, col3 = st.beta_columns([5, 5, 5])
        with col1:
            st.dataframe(Goals.loc[Goals['P2Name'] == PlayerSelect].
                         append(Goals.loc[Goals['P3Name'] == PlayerSelect]).
                         append(Goals.loc[Goals['P4Name'] == PlayerSelect]), height=500)
        with col2:
            st.dataframe(Shots.loc[Shots['P2Name'] == PlayerSelect], height=500)

    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16, col17, col18, \
    col19, col20, col21, col22, col23, col24, col25, col26, col27, col28, col29, col30, col31 = st.beta_columns(31)
    with col1:
        st.image('./TeamLogos/ANA.png', use_column_width=True)
    with col2:
        st.image('./TeamLogos/ARI.png', use_column_width=True)
    with col3:
        st.image('./TeamLogos/BOS.png', use_column_width=True)
    with col4:
        st.image('./TeamLogos/BUF.png', use_column_width=True)
    with col5:
        st.image('./TeamLogos/CAR.png', use_column_width=True)
    with col6:
        st.image('./TeamLogos/CBJ.png', use_column_width=True)
    with col7:
        st.image('./TeamLogos/CGY.png', use_column_width=True)
    with col8:
        st.image('./TeamLogos/CHI.png', use_column_width=True)
    with col9:
        st.image('./TeamLogos/COL.png', use_column_width=True)
    with col10:
        st.image('./TeamLogos/DAL.png', use_column_width=True)
    with col11:
        st.image('./TeamLogos/DET.png', use_column_width=True)
    with col12:
        st.image('./TeamLogos/EDM.png', use_column_width=True)
    with col13:
        st.image('./TeamLogos/FLA.png', use_column_width=True)
    with col14:
        st.image('./TeamLogos/LAK.png', use_column_width=True)
    with col15:
        st.image('./TeamLogos/MIN.png', use_column_width=True)
    with col16:
        st.image('./TeamLogos/MTL.png', use_column_width=True)
    with col17:
        st.image('./TeamLogos/NJD.png', use_column_width=True)
    with col18:
        st.image('./TeamLogos/NSH.png', use_column_width=True)
    with col19:
        st.image('./TeamLogos/NYI.png', use_column_width=True)
    with col20:
        st.image('./TeamLogos/NYR.png', use_column_width=True)
    with col21:
        st.image('./TeamLogos/OTT.png', use_column_width=True)
    with col22:
        st.image('./TeamLogos/PHI.png', use_column_width=True)
    with col23:
        st.image('./TeamLogos/PIT.jpg', use_column_width=True)
    with col24:
        st.image('./TeamLogos/SJS.png', use_column_width=True)
    with col25:
        st.image('./TeamLogos/STL.png', use_column_width=True)
    with col26:
        st.image('./TeamLogos/TBL.png', use_column_width=True)
    with col27:
        st.image('./TeamLogos/TOR.png', use_column_width=True)
    with col28:
        st.image('./TeamLogos/VAN.png', use_column_width=True)
    with col29:
        st.image('./TeamLogos/VGK.png', use_column_width=True)
    with col30:
        st.image('./TeamLogos/WPG.png', use_column_width=True)
    with col31:
        st.image('./TeamLogos/WSH.png', use_column_width=True)

if(Page == 'Predictive Analytics'):

    Skaters = NHLRoster.loc[NHLRoster['PosName'] != 'Goalie'].reset_index()

    PlayerSelect = st.sidebar.selectbox('Select a Player',
                                        Skaters['FullName'])
    PlayerID = NHLRoster.loc[NHLRoster['FullName'] == PlayerSelect]['ID'].values[0]

    st.header("Predicted Points for " + PlayerSelect)

    pd.options.display.max_columns = None
    Seasons = ['20152016', '20162017', '20172018', '20182019', '20192020', ]

    SkaterStatistics = pd.DataFrame()
    for Season in tqdm(Seasons):
        intCounter = 0

        print("https://statsapi.web.nhl.com/api/v1/people/" + str(
            PlayerID) + "/stats?stats=statsSingleSeason&season=" + Season)
        SourceData = pd.DataFrame.from_dict(requests.get(url="https://statsapi.web.nhl.com/api/v1/people/" + str(
            PlayerID) + "/stats?stats=statsSingleSeason&season=" + Season).json()).to_dict()['stats'][0]['splits']
        try:
            SD = pd.DataFrame.from_dict(SourceData)['stat']
            SD = SD.to_dict()[0]

            CurrentPlayer = StringIO("TOI" + str(Season) +
                                     '; A' + str(Season) +
                                     '; G' + str(Season) +
                                     '; P' + str(Season) +
                                     '; PIM' + str(Season) +
                                     '; Shots' + str(Season) +
                                     '; Games' + str(Season) +
                                     '; Hits' + str(Season) +
                                     '; PPG' + str(Season) +
                                     '; PPP' + str(Season) +
                                     '; PPTOI' + str(Season) +
                                     '; EVTOI' + str(Season) +
                                     '; PIM2' + str(Season) +
                                     '; FaceOffPCT' + str(Season) +
                                     '; ShtPercent' + str(Season) +
                                     '; GWG' + str(Season) +
                                     '; OTG' + str(Season) +
                                     '; SHG' + str(Season) +
                                     '; SHP' + str(Season) +
                                     '; SHTOI' + str(Season) +
                                     '; Blocks' + str(Season) +
                                     '; PlusMinus' + str(Season) +
                                     '; Shifts' + str(Season) +
                                     '; TOIperGame' + str(Season) +
                                     '; EVTOIperGame' + str(Season) +
                                     '; SHTOIperGame' + str(Season) +
                                     '; PPTOIperGame' + str(Season) + """
                                     """ +
                                     str(SD['timeOnIce']) + "; " +
                                     str(SD['assists']) + "; " +
                                     str(SD['goals']) + "; " +
                                     str(SD['points']) + "; " +
                                     str(SD['pim']) + "; " +
                                     str(SD['shots']) + "; " +
                                     str(SD['games']) + "; " +
                                     str(SD['hits']) + "; " +
                                     str(SD['powerPlayGoals']) + "; " +
                                     str(SD['powerPlayPoints']) + "; " +
                                     str(SD['powerPlayTimeOnIce']) + "; " +
                                     str(SD['evenTimeOnIce']) + "; " +
                                     str(SD['penaltyMinutes']) + "; " +
                                     str(SD['faceOffPct']) + "; " +
                                     str(SD['shotPct']) + "; " +
                                     str(SD['gameWinningGoals']) + "; " +
                                     str(SD['overTimeGoals']) + "; " +
                                     str(SD['shortHandedGoals']) + "; " +
                                     str(SD['shortHandedPoints']) + "; " +
                                     str(SD['shortHandedTimeOnIce']) + "; " +
                                     str(SD['blocked']) + "; " +
                                     str(SD['plusMinus']) + "; " +
                                     str(SD['shifts']) + "; " +
                                     str(SD['timeOnIcePerGame']) + "; " +
                                     str(SD['evenTimeOnIcePerGame']) + "; " +
                                     str(SD['shortHandedTimeOnIcePerGame']) + "; " +
                                     str(SD['powerPlayTimeOnIcePerGame']))
            CurrentPlayer = pd.read_csv(CurrentPlayer, sep=";")

            SkaterStatistics = pd.concat([SkaterStatistics, CurrentPlayer], axis=1)
            #         SkaterStatistics = SkaterStatistics.append(CurrentPlayer, ignore_index=True)
            #         SkaterStatistics = SkaterStatistics.merge(CurrentPlayer, on='ID', how='outer')
            intCounter += 1
        except:
            # print("Missing Stats for Player " + PlayerLink + " " + SkaterRoster['FullName'][intCounter])
            intCounter += 1

    SS = SkaterStatistics

    Stats = ['TOI', 'Assists', 'Goals', 'Points', 'PIMS', 'Shots', 'Games', 'Hits', 'PPG', 'PPP', 'PPTOI', 'ESTOI',
             'PIM2', 'FOPct', 'ShtPct',
             'GWG', 'OTG', 'SHG', 'SHP', 'SHTOI', 'Blocks', 'PlusMinus', 'Shifts', 'TOIperGame', 'EVTOIperGame',
             'SHTOIperGame', 'PPTOIperGame']

    NewColumnNames = (
            (pd.Series(Stats) + "LastYear").tolist() +
            (pd.Series(Stats) + "2Years").tolist() +
            (pd.Series(Stats) + "3Years").tolist() +
            (pd.Series(Stats) + "4Years").tolist() +
            (pd.Series(Stats) + "5Years").tolist()
    )

    SS.columns = NewColumnNames
    SS = SS.drop(['TOILastYear', 'TOI2Years', 'TOI3Years', 'TOI4Years', 'TOI5Years'], axis=1)

    SS = SS.drop([
        'PIMSLastYear', 'HitsLastYear', 'PIM2LastYear',
        'FOPctLastYear', 'ShtPctLastYear', 'OTGLastYear', 'SHGLastYear', 'SHPLastYear',
        'SHTOILastYear', 'BlocksLastYear', 'PlusMinusLastYear',
        'EVTOIperGameLastYear', 'SHTOIperGameLastYear', 'GamesLastYear', 'ESTOILastYear',
        'ShiftsLastYear', 'TOIperGameLastYear', 'PPGLastYear', 'GWGLastYear',

        'PIMS2Years', 'Hits2Years', 'PIM22Years',
        'FOPct2Years', 'ShtPct2Years', 'OTG2Years', 'SHG2Years', 'SHP2Years',
        'SHTOI2Years', 'Blocks2Years', 'PlusMinus2Years',
        'EVTOIperGame2Years', 'SHTOIperGame2Years', 'Games2Years', 'ESTOI2Years',
        'Shifts2Years', 'TOIperGame2Years', 'PPG2Years', 'GWG2Years',

        'PIMS3Years', 'Hits3Years', 'PIM23Years',
        'FOPct3Years', 'ShtPct3Years', 'OTG3Years', 'SHG3Years', 'SHP3Years',
        'SHTOI3Years', 'Blocks3Years', 'PlusMinus3Years',
        'EVTOIperGame3Years', 'SHTOIperGame3Years', 'Games3Years', 'ESTOI3Years',
        'Shifts3Years', 'TOIperGame3Years', 'PPG3Years', 'GWG3Years',

        'PIMS4Years', 'Hits4Years', 'PIM24Years',
        'FOPct4Years', 'ShtPct4Years', 'OTG4Years', 'SHG4Years', 'SHP4Years',
        'SHTOI4Years', 'Blocks4Years', 'PlusMinus4Years',
        'EVTOIperGame4Years', 'SHTOIperGame4Years', 'Games4Years', 'ESTOI4Years',
        'Shifts4Years', 'TOIperGame4Years', 'PPG4Years', 'GWG4Years',

        'PIMS5Years', 'Hits5Years', 'PIM25Years',
        'FOPct5Years', 'ShtPct5Years', 'OTG5Years', 'SHG5Years', 'SHP5Years',
        'SHTOI5Years', 'Blocks5Years', 'PlusMinus5Years',
        'EVTOIperGame5Years', 'SHTOIperGame5Years', 'Games5Years', 'ESTOI5Years',
        'Shifts5Years', 'TOIperGame5Years', 'PPG5Years', 'GWG5Years', ], axis=1)

    Player = 0
    SS['Years'] = 0
    if SS['PointsLastYear'][Player] > 0:
        if SS['Points2Years'][Player] > 0:
            if SS['Points3Years'][Player] > 0:
                if SS['Points4Years'][Player] > 0:
                    if SS['Points5Years'][Player] > 0:
                        SS['Years'][Player] = 5
                    else:
                        SS['Years'][Player] = 4
                else:
                    SS['Years'][Player] = 3
            else:
                SS['Years'][Player] = 2
        else:
            SS['Years'][Player] = 1
    else:
        SS['Years'][Player] = 0

    # print(SS)

    SS = SS.replace({":": "."}, regex=True)
    SS['PPTOILastYear'] = SS['PPTOILastYear'].astype(float)

    SS['PPTOIperGameLastYear'] = SS['PPTOIperGameLastYear'].astype(float)
    SS['PPTOI2Years'] = SS['PPTOI2Years'].astype(float)
    SS['PPTOIperGame2Years'] = SS['PPTOIperGame2Years'].astype(float)
    SS['PPTOI3Years'] = SS['PPTOI3Years'].astype(float)
    SS['PPTOIperGame3Years'] = SS['PPTOIperGame3Years'].astype(float)
    SS['PPTOI4Years'] = SS['PPTOI4Years'].astype(float)
    SS['PPTOIperGame4Years'] = SS['PPTOIperGame4Years'].astype(float)
    SS['PPTOI5Years'] = SS['PPTOI5Years'].astype(float)
    SS['PPTOIperGame5Years'] = SS['PPTOIperGame5Years'].astype(float)

    Maxs = [83.00, 60.00, 112.00, 398.00, 44.00, 426.48, 5.25, 83.00, 60.00, 113.00, 528.00, 46.00, 455.32, 6.14, 83.00,
            65.00, 113.00, 528.00,
            46.00, 470.45, 6.14, 92.00, 65.00, 120.00, 528.00, 61.00, 602.57, 7.21, 96.00, 65.00, 125.00, 528.00, 61.00,
            602.57, 7.21, 5.00]

    normalizedSS = SS / Maxs


    def coeff_determination(y_true, y_pred):
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - SS_res / (SS_tot + K.epsilon()))


    model = load_model('best_model.h5', custom_objects={'coeff_determination': coeff_determination})

    Prediction = model.predict(normalizedSS)

    Prediction = round(Prediction.ravel()[0])

    st.write(Prediction)