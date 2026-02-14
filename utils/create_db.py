from sqlalchemy import Column, String, Integer, Float, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

#--------------------------------------------------------------
# Table teams
class Team(Base):
    __tablename__ = "teams"

    Code = Column(String, primary_key=True)
    TeamName = Column(String, nullable=False)

    players = relationship('Player', back_populates='team')


#---------------------------------------------------------------
# Table description
class Description(Base): 
    __tablename__ = "descriptions" 

    Data = Column(String, primary_key=True)
    Description = Column(String, nullable=False)


#---------------------------------------------------------------
# Table principale
class Player(Base):
    __tablename__ = 'players'

    id = Column(Integer, primary_key=True, autoincrement=True)

    Player = Column(String)
    Team = Column(String, ForeignKey("teams.Code"))
    Age = Column(Integer)
    GP = Column(Integer)
    W = Column(Integer)
    L = Column(Integer)
    Min = Column(Float)
    PTS = Column(Integer)
    FGM = Column(Integer)
    FGA = Column(Integer)
    FGpct = Column(Float)
    _15_00_00 = Column(Integer)
    _3PA = Column(Integer)
    _3Ppct = Column(Float)
    FTM = Column(Integer)
    FTA = Column(Integer)
    FTpct = Column(Float)
    OREB = Column(Integer)
    DREB = Column(Integer)
    REB = Column(Integer)
    AST = Column(Integer)
    TOV = Column(Integer)
    STL = Column(Integer)
    BLK = Column(Integer)
    PF = Column(Integer)
    FP = Column(Integer)
    DD2 = Column(Integer)
    TD3 = Column(Integer)
    plus_minus = Column(Float)
    OFFRTG = Column(Float)
    DEFRTG = Column(Float)
    NETRTG = Column(Float)
    ASTpct = Column(Float)
    AST_TO = Column(Float)
    AST_RATIO = Column(Float)
    OREBpct = Column(Float)
    DREBpct = Column(Float)
    REBpct = Column(Float)
    TO_RATIO = Column(Float)
    EFGpct = Column(Float)
    TSpct = Column(Float)
    USGpct = Column(Float)
    PACE = Column(Float)
    PIE = Column(Float)
    POSS = Column(Integer)

    team = relationship('Team', back_populates='players')


#---------------------------------------------------------------
# creation de la base sqlite
engine = create_engine("sqlite:///../data/nba.db")

Base.metadata.create_all(engine)

print("Base SQLite créée avec succès.")
