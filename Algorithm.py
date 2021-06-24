import numpy as np

def RotateVector(Vector):
    return np.array([Vector[0]**2 - Vector[1]**2, 2*Vector[0]*Vector[1]])
def RotateVectors(Vectors):
    RotatedVectors = np.zeros(Vectors.shape)
    RotatedVectors[:,0] = Vectors[:,0]**2 - Vectors[:,1]**2
    RotatedVectors[:,1] = 2*Vectors[:,0]*Vectors[:,1]
    return RotatedVectors

def NonZeroNumber(x):
    return x + int(x == 0)
def NonZeroArray(A):
    return A + np.array(A == 0, dtype = int)
def NonZero(x):
    if type(x) == n.ndarray:
        return NonZeroArray(x)
    else:
        return NonZeroNumber(x)

class TrackerModule:
    def __init__(self):
        '''
        Class handling event-based tracking via algorithm MASCOT.
        Input is an event tuple ((x,y), t, p) for an event
        Output is a list of tuples ((x_T, y_T), ID_T), x_T and y_T being the coordinates of the tracker on screen, and ID_T is the number for that particular tracker.

        We left the most interesting parameters here, with their expected effects. However, most of them are set by default on the most effective values for tracking. 
        The most important value to be changed is the tracker diameter, as it has the most impact depending on the scene.
        '''
        self._ScreenSize = np.array([346, 260])
        self._DetectorPointsGridMax = (10, 8)                                       # (Int, Int). Reference grid to place trackers, capping the discrete grid made by dividing the screen with _TrackerDiameter. Default : (10,8), i.e 80 trackers
        self._DetectorMinRelativeStartActivity = 2.                                 # Float. Activity of a tracker needed to activate, divided by _TrackerDiameter. Low values are noise sensitive, while high value leave tracker in IDLE state. Default : 2.0
        self._TrackerDiameter = 25.                                                 # Int. Size of a tracker's ROI radius. Usually 30 for qVGA, 25 for 346x260.
        self._NPixelsEdge = 1.5                                                     # Float. Defines the average length over which the different estimator are decayed.
        self._RelativeActivityCap = 2.0                                             # Float. Boundary for maximum activity a tracker can reach, taking into account _TrackerDiameter and _NPixelsEdge. Default : 2.0
        self._DetectorDefaultSpeedTrigger = 5.                                      # Float. Default speed trigger, giving a lower bound for speeds in the scene, and higher boundary in time constants. Default : 5px/s
        self._TrackerMinDeathActivity = 0.1                                         # Float. Low boundary for a tracker to survive divided by _TrackerDiameter. Must be lower than _DetectorMinRelativeStartActivity. Default : 0.1

        self._ShapeMaxOccupancy = 0.3                                               # Float. Maximum occupancy of the tracker ROI. Default : 0.3
        self._ClosestEventProximity = 2.5                                           # Float. Maximum distance for an event to be considered in the simple flow computation. Default : 2.5
        self._MaxConsideredNeighbours = 20                                          # Int. Maximum number of considered events in the simple flow computation. Higher values slighty increase flow quality, but highly increase computation time. Default : 20

        self._TrackerOffCenterThreshold = 0.4                                       # Float. Minimum distance of to consider a tracker off-centered relative to _TrackerDiameter/2. Default : 0.4
        self._TrackerMeanPositionDisplacementFactor = 0.8                           # Float. Amount of correction due to recentering process. Default : 0.8

        self._TrackerConvergenceThreshold = 0.5                                     # Float. Amount of correction relative to activity allowing for a converged feature. Default : 0.5
        self._TrackerConvergenceHysteresis = 0.1                                    # Float. Hysteresis value of _TrackerConvergenceThreshold. Default : 0.1

        self._TrackerApertureIssueThreshold = 0.55                                  # Float. Amount of aperture scalar value by speed relative to correction activity. Default : 0.55
        self._TrackerApertureIssueHysteresis = 0.05                                 # Float. Hysteresis value of _TrackerApertureIssueThreshold. Default : 0.05

        self._LockupMaximumSpread = 0.7                                             # Float. Maximum occupancy of the shape defined by the  projected events at lockup. This protects against highly textured ROIs and stabilization failures. >=1 disables this filter. Default : 0.7
        self._TrackerLockingMaxRecoveryBuffer = 100                                 # Int. Size of the buffer storing events while locked, used in case of feature loss. Greater values increase computationnal cost. Default : 100
        self._TrackerLockedRelativeCorrectionsFailures = 0.7                        # Float. Minimum correction activity relative to tracker activity for a lock to remain. Assesses for shape loss. Default : 0.7
        self._TrackerLockedRelativeCorrectionsHysteresis = 0.05                     # Float. Hysteresis value of _TrackerLockedRelativeCorrectionsFailures. Default : 0.05
        self._TrackerDisengageActivityThreshold = 0.5                               # Float. Activity ratio below which a tracker disengages, due to fast deceleration, inducing a lack of event. Default : 0.5
        self._TrackerDisengageActivityHysteresis = 0.1                              # Float. Threshold associated to previous value

        self.TrackerMaxEventsBuffer = np.pi*(self._TrackerDiameter/2)**2 * 4 * self._ShapeMaxOccupancy
        self.LocalEdgeRadius = self._ClosestEventProximity * 1.1
        self.LocalEdgeNumberOfEvents = int(4*self.LocalEdgeRadius)
        self.MinConsideredNeighbours = int(self._ClosestEventProximity * 2 * 1.5)  

        L_X, L_Y = self._ScreenSize
        self.Trackers = []
        self.AliveTrackers = []
        self.JustDeadTrackers = [] # Used to store trackers that just died, in order to have their last values recorded.
        self.RecordedTrackers = []
        self.StartTimes = []
        self.DeathTimes = []

        d_x = self._TrackerDiameter
        d_y = self._TrackerDiameter
        N_X = int(L_X / d_x)
        N_Y = int(L_Y / d_y)

        if N_X > self._DetectorPointsGridMax[0]:
            N_X = self._DetectorPointsGridMax[0]
            d_x = L_X / N_X
        if N_Y > self._DetectorPointsGridMax[1]:
            N_Y = self._DetectorPointsGridMax[1]
            d_y = L_Y / N_Y

        r_X = (L_X - (N_X - 1) * d_x)/2
        r_Y = (L_Y - (N_Y - 1) * d_y)/2

        self._TrackerDefaultTimeConstant = self._NPixelsEdge / self._DetectorDefaultSpeedTrigger
        
        self.DetectorInitialPositions = []
        for nX in range(N_X):
            for nY in range(N_Y):
                self.DetectorInitialPositions += [np.array([r_X + nX * d_x, r_Y + nY * d_y])]

        for ID, InitialPosition in enumerate(self.DetectorInitialPositions):
            self._AddTracker(InitialPosition)

        self._DetectorMinActivityForStart = self._TrackerDiameter * self._DetectorMinRelativeStartActivity

    def OnEvent(self, event):
        self.NewTrackersAsked = 0
        TrackersData = []
        for Tracker in self.AliveTrackers:
            TrackerID = Tracker.ID
            TrackerData = Tracker.RunEvent(event)
            if TrackerData and self.TrackerEventCondition(Tracker):
                TrackersData += TrackerData

        if self.NewTrackersAsked:
            self._PlaceNewTrackers()
            self.NewTrackersAsked = 0

        return TrackersData

    def TrackerEventCondition(self, Tracker):
        return Tracker.State.Status == StateClass._STATUS_LOCKED

    def _KillTracker(self, Tracker, t, Reason=''):
        print("Tracker {0} died".format(Tracker.ID) + int(bool(Reason)) * (" ("+Reason+")"))
        Tracker.State.SetStatus(Tracker.State._STATUS_DEAD)
        self.DeathTimes[Tracker.ID] = t
        self.AliveTrackers.remove(Tracker)
        self.JustDeadTrackers += [Tracker]

        if self._DetectorAutoRestart:
            self.NewTrackersAsked += 1

    def _PlaceNewTrackers(self):
        for NewID in range(self.NewTrackersAsked):
            SelectedPositionID = None
            MaxDistance = 0
            
            for InitialPositionID, InitialPosition in enumerate(self.DetectorInitialPositions):
                InitialPositionMinDistance = np.inf
                for Tracker in self.AliveTrackers:
                    InitialPositionMinDistance = min(InitialPositionMinDistance, np.linalg.norm(InitialPosition - Tracker.Position[:2]))
                if InitialPositionMinDistance >= MaxDistance:
                    MaxDistance = InitialPositionMinDistance
                    SelectedPositionID = InitialPositionID
            self._AddTracker(self.DetectorInitialPositions[SelectedPositionID])

    def _AddTracker(self, InitialPosition):
        ID = len(self.Trackers)
        NewTracker = TrackerClass(self, ID, InitialPosition)
        self.Trackers += [NewTracker]
        self.AliveTrackers += [NewTracker]
        self.StartTimes += [None]
        self.DeathTimes += [None]

class StateClass:
    _STATUS_DEAD = 0
    _STATUS_IDLE = 1
    _STATUS_STABILIZING = 2
    _STATUS_CONVERGED = 3
    _STATUS_LOCKED = 4

    # Properties can stack, so they are given as powers on binary number
    _PROPERTY_APERTURE = 0
    _PROPERTY_OFFCENTERED = 1
    _PROPERTY_DISENGAGED = 2
    
    _StatusesNames = {_STATUS_DEAD: 'Dead', _STATUS_IDLE:'Idle', _STATUS_STABILIZING:'Stabilizing', _STATUS_CONVERGED:'Converged', _STATUS_LOCKED: 'Locked'} # Statuses names with associated int values.
    _PropertiesNames = {0: 'None', 1: 'Aperture issue', 2:'OffCentered', 3:'Disengaged'} # Properties names with associated int values.

    def __init__(self):
        self.Status = self._STATUS_IDLE
        self.ApertureIssue = False
        self.OffCentered = False
        self.Disengaged = False
    def __repr__(self):
        return str(self.Value)

    @property
    def Properties(self):
        return (self.OffCentered << self._PROPERTY_OFFCENTERED | self.ApertureIssue << self._PROPERTY_APERTURE | self.Disengaged << self._PROPERTY_DISENGAGED)
    @property
    def Value(self):
        return (self.Status, self.Properties)
    def SetStatus(self, Value):
        self.Status = Value
    def __eq__(self, RHS):
        return self.Status == RHS
    @property
    def Idle(self):
        return self.Status == self._STATUS_IDLE
    @property
    def Stabilizing(self):
        return self.Status == self._STATUS_STABILIZING
    @property
    def Converged(self):
        return self.Status == self._STATUS_CONVERGED
    @property
    def Locked(self):
        return self.Status == self._STATUS_LOCKED
    @property
    def Dead(self):
        return self.Status == self._STATUS_DEAD

class LockClass:
    def __init__(self, Time, TrackerActivity, FlowActivity, Events):
        self.Time = Time
        self.TrackerActivity = TrackerActivity
        self.FlowActivity = FlowActivity
        self.Events = Events
        self.ReleaseTime = None

class EstimatorTemplate:
    def __init__(self):
        self.LastUpdate = -np.inf
        self.W = 0
        self._DecayingVars = ['W']
        self._GeneralVars = []
    def __iter__(self):
        self._IterVar = 0
        return self
    def __next__(self):
        if self._IterVar >= len(self._DecayingVars):
            raise StopIteration
        self._IterVar += 1
        Var = self.__dict__[self._DecayingVars[self._IterVar-1]]
        if type(Var)==np.ndarray:
            return np.array(Var)
        else:
            return type(Var)(Var)
    def AddDecayingVar(self, VarName, Dimension = 1, GeneralVar = False):
        if Dimension == 1:
            self.__dict__['_'+VarName] = 0.
            if GeneralVar:
                self.__dict__[VarName] = 0.
                self._GeneralVars += [VarName]
        else:
            self.__dict__['_'+VarName] = np.zeros(Dimension)
            if GeneralVar:
                self.__dict__[VarName] = np.zeros(Dimension)
                self._GeneralVars += [VarName]
        self._DecayingVars += ['_'+VarName]
    def RecoverGeneralData(self):
        if self.W:
            for VarName in self._GeneralVars:
                self.__dict__[VarName] = self.__dict__['_'+VarName] / NonZeroNumber(self.W)
    def EstimatorStep(self, newT, Tau, WeightIncrease = 1):
        DeltaUpdate = newT - self.LastUpdate
        self.LastUpdate = newT
        Decay = np.e**(-DeltaUpdate/Tau)
        for VarName in self._DecayingVars:
            self.__dict__[VarName] *= Decay
        self.W += WeightIncrease

class DynamicsEstimatorClass(EstimatorTemplate):
    def __init__(self, Radius):
        EstimatorTemplate.__init__(self)
        self._Radius = Radius
        self._Dim = 3

        self._UpToDateMatrix = False
        self._LowerInvertLimit = (Radius/2)**(self._Dim - 2)*2 * 0.5**self._Dim

        self._M = np.zeros((self._Dim, self._Dim))
        self.MDet = 0.
        self._InvM = np.zeros((self._Dim, self._Dim))

        self.AddDecayingVar('Es', self._Dim, GeneralVar = True)
        self.AddDecayingVar('Ed', self._Dim, GeneralVar = True)

        self.AddDecayingVar('X', 2, GeneralVar = True)
        self.AddDecayingVar('r2', 1, GeneralVar = True)

        for ConvergingSum in ['Scc', 'Sxxss', 'Syycc']:
            self.AddDecayingVar(ConvergingSum)
        self.AddDecayingVar('Rcs', 1)
        self.AddDecayingVar('RXcc', 2)
        self.AddDecayingVar('RXcs', 2)

    def _EstimatorVariation(self, Observable, x, y):
        Var = np.zeros(self._Dim)
        Var[:2] = Observable
        Var[2] = Observable[1]*x-Observable[0]*y
        return Var
    def _EstimatorShift(self, Estimator, dx, dy):
        return np.array([0., 0., Estimator[1]*dx - Estimator[0]*dy # The minus sign will be placed outside

    def TrackerShift(self, PositionVariation): # An event initially in (x,y) is now in (x-dx, y-dy)
        dx, dy = PositionVariation
        N2PositionVariation = (PositionVariation**2).sum()
        self._Es -= self._EstimatorShift(self._Es, dx, dy) # W is hidden in the first two  terms
        self._Ed -= self._EstimatorShift(self._Ed, dx, dy)

        self._X -= self.W*PositionVariation # If X > 0, the events are too far right in tracker frame. The tracker compensates by moving right, thus PositionVariation > 0, X' is thus lowered.
        self._r2 -= 2*(self._X*PositionVariation).sum() + self.W*N2PositionVariation # We can use the new _X coordinates to simplify the process

        # Scc, Sss and Rcs are unchanged since they only use optical flow orientations
    
        self._RXcs  -= self._Rcs * PositionVariation
        self._RXcc  -= self._Scc * PositionVariation

        _RXss = self._X - self._RXcc
        _Sss  = self.W  - self._Scc
        self._Sxxss -= 2* _RXss[0] * dx + _Sss * dx**2
        self._Syycc -= 2* self._RXcc[1] * dy + self._Scc * dy**2

    def AddData(self, t, TrackerRelativeLocation, Flow, Displacement, Tau):
        F2 = Flow**2
        N2 = F2.sum()
        if N2 == 0:
            return
        self.EstimatorStep(t, Tau)

        self._X += TrackerRelativeLocation
        x, y = TrackerRelativeLocation  # If using tracker center

        xx, yy = x**2, y**2
        xy = x*y

        N = np.sqrt(N2)
        c , s  = Flow/N
        cc, ss = F2/N2
        cs = c*s

        self._Es += self._EstimatorVariation(Flow, x, y)
        self._Ed += self._EstimatorVariation(Displacement, x, y)

        self._r2 += xx+yy

        self._Scc += cc
        self._Sxxss += xx*ss
        self._Syycc += yy*cc

        self._Rcs  += cs
        self._RXcs += np.array([x,y])*cs
        self._RXcc += np.array([x,y])*cc

        self._UpToDateMatrix = False

    def _GetInverseMatrix(self):
        if self._UpToDateMatrix:
            return (abs(self.MDet) > self._LowerInvertLimit)

        RXss = self._X - self._RXcc

        self._M[0,0] = self._Scc
        self._M[1,1] = self.W - self._Scc
        self._M[2,2] = self._Sxxss + self._Syycc - 2*self._Rxycs

        self._M[1,0] = self._M[0,1] = self._Rcs
        self._M[2,0] = self._M[0,2] = self._RXcs[0] - self._RXcc[1]
        self._M[2,1] = self._M[1,2] = (RXss[0]) - self._RXcs[1]

        self.MDet = np.linalg.det(self._M) / (NonZeroNumber(self.W) ** self._Dim)
        self._UpToDateMatrix = True
        if abs(self.MDet) > self._LowerInvertLimit: 
            self._InvM = np.linalg.inv(self._M)
            return True
        else:
            return False

    @property
    def Speed(self):
        return self.GetSpeed()

    def GetSpeed(self):
        if self._GetInverseMatrix():
            return self._InvM.dot(self._Es)
        else:
            SimpleTranslation = self._Es / NonZeroNumber(self.W)
            SimpleTranslation[2] = 0
            return SimpleTranslation

    @property
    def Displacement(self):
        return self.GetDisplacement()
    def GetDisplacement(self):
        if self._GetInverseMatrix():
            return self._InvM.dot(self._Ed)
        else:
            SimpleTranslation = self._Ed / NonZeroNumber(self.W)
            SimpleTranslation[2] = 0
            return SimpleTranslation

class SpeedConvergenceEstimatorClass(EstimatorTemplate):
    def __init__(self, Type):
        EstimatorTemplate.__init__(self)
        self.AddDecayingVar('Epsilon', Dimension = 1, GeneralVar = True)
        self.AddDecayingVar('Sigma2', Dimension = 1, GeneralVar = True)

    @property
    def Value(self):
        return self.GetValue()

    def AddData(self, t, ProjectionError, Tau):
        NProjectionError = np.linalg.norm(ProjectionError)
        self.EstimatorStep(t, Tau)
        self._Epsilon += NProjectionError
        self._Sigma2 += (NProjectionError - self._Epsilon/self.W)**2
    def GetValue(self):
        self.RecoverGeneralData()
        return self.Epsilon, np.sqrt(self.Sigma2)

class ApertureEstimatorClass(EstimatorTemplate):
    def __init__(self):
        EstimatorTemplate.__init__(self)

        self.AddDecayingVar('Vector', 2, GeneralVar = True)
        self.AddDecayingVar('Deviation', 1, GeneralVar = True)

    def AddData(self, t, LocalVector, Tau):
        N = (LocalVector**2).sum()
        if N == 0:
            return
        self.EstimatorStep(t, Tau, WeightIncrease = np.sqrt(N))

        self._Vector += LocalVector
        self._Deviation += np.linalg.norm(LocalVector - self._Vector / NonZeroNumber(self.W))
    @property
    def Value(self):
        self.RecoverGeneralData()
        return np.linalg.norm(self.Vector)

class DynamicsModifierClass:
    _SaveData = False
    def __init__(self, Tracker):
        self.Tracker = Tracker
        self.PositionMods = {}
        self.SpeedMods = {}
        self.BoolFactors = {"Speed":{}, "Position":{}}
        self._NoModValue = np.array([0., 0., 0.])

    def AddModifier(self, Name, AffectSpeed = [False, False], AffectPosition = [False, False]):
        if True in AffectSpeed:
            self.SpeedMods[Name] = self._NoModValue
            self.BoolFactors["Speed"][Name] = np.array([AffectSpeed[0], AffectSpeed[0], AffectSpeed[1]])
        if True in AffectPosition:
            self.PositionMods[Name] = self._NoModValue
            self.BoolFactors["Position"][Name] = np.array([AffectPosition[0], AffectPosition[0], AffectPosition[1]])
    def Compile(self):
        for Origin, Value in self.SpeedMods.items():
            self.Tracker.Speed += Value
            self.SpeedMods[Origin] = self._NoModValue
        for Origin, Value in self.PositionMods.items():
            self.Tracker.Position += Value
            self.PositionMods[Origin] = self._NoModValue

    def ModSpeed(self, Origin, Value):
        self.SpeedMods[Origin] = Value * self.BoolFactors["Speed"][Origin]
    def ModPosition(self, Origin, Value):
        self.PositionMods[Origin] = Value * self.BoolFactors["Position"][Origin]

class TrackerClass:
    _Modifiers = {'Speed':((False, False), (True, True)), # Speed modification (T,R), Position modification (T,R)
                  'Flow': ((True, True),   (True, True)),
                  'MeanPos': ((False, False), (True, False)),
                  'Disengage': ((True, True), (True, True))}
    def __init__(self, TrackerManager, ID, InitialPosition):
        self.TM = TrackerManager
        self.ID = ID

        self.State = StateClass()
        self.Lock = None

        self.Position = np.array(list(InitialPosition) + [0.], dtype = float) # We now include all 3 parameters of the tracker in a single variable. Easier resolution and code compacity
        self.Speed    = np.array([0., 0., 0.], dtype = float)
        # self.Position = [tx, ty, theta]
        self.Radius = self.TM._TrackerDiameter / 2
        self.SquaredRadius = self.Radius**2

        self.TimeConstant = self.TM._TrackerDefaultTimeConstant
        self.EdgeBinTC = self.TimeConstant * self.TM._EdgeBinRatio
        
        self.ProjectedEvents = []

        self.LastUpdate = 0.
        self.LastValidFlow = 0.
        self.LastRecenter = 0.

        self.TrackerActivity = 0.
        self.FlowActivity = 0.
        self.DynamicsEstimator = DynamicsEstimatorClass(self.Radius)
        self.ApertureEstimator = ApertureEstimatorClass()
        self.SpeedConvergenceEstimator = SpeedConvergenceEstimatorClass()

        self.MeanPosCorrection = np.array([0., 0.]) # Computed in tracker frame

        self.DynamicsModifier = DynamicsModifierClass(self)
        for Modifier, (AffectSpeed, AffectPosition) in self._Modifiers.items():
            self.DynamicsModifier.AddModifier(Modifier, AffectSpeed = AffectSpeed, AffectPosition = AffectPosition)

    def UpdateWithEvent(self, event):
        xy, t, p = event
        DeltaUpdate = t - self.LastUpdate
        self.LastUpdate = t
        Decay = np.e**(-DeltaUpdate / self.TimeConstant)
        
        if not self.State.Disengaged:
            self.DynamicsModifier.ModPosition('Speed', self.Speed * DeltaUpdate)
            self.DynamicsModifier.Compile()
            self.Position[3] = min(max(self.Position[3], 0.1), 10.)
        else:
            self.DynamicsModifier.ModSpeed('Disengage', self.Speed * (Decay-1)) 
            self.DynamicsModifier.Compile()
            self._UpdateTC()

        self.TrackerActivity *= Decay
        self.FlowActivity *= Decay
        self.MeanPosCorrection *= Decay

        if self.State.Idle:
            return True
        if not self.State.Disengaged and ((self.Position[:2] < 0).any() or (self.Position[:2] >= np.array(self.TM._LinkedMemory.STContext.shape[:2])).any()): # out of bounds, cannot happen if disengaged
            if self.Lock:
                self.Unlock('out of bounds')
            self.TM._KillTracker(self, t, Reason = "out of bounds")
            return False
        if self.State.Locked:
            if not self.State.Disengaged and self.TrackerActivity < self.TM._TrackerDiameter * (self.TM._TrackerDisengageActivityThreshold - self.TM._TrackerDisengageActivityHysteresis):
                self.Disengage(t)
                return True
        if self.TrackerActivity < self.TM._TrackerDiameter * self.TM._TrackerMinDeathActivity: # Activity remaining too low
            if self.Lock:
                self.Unlock('low activity')
            self.TM._KillTracker(self, t, "low activity")
            return False
        return True

    def RunEvent(self, event):
        if not self.UpdateWithEvent(event): # We update the position and activity. They do not depend on the event location. False return means the tracker has died : out of screen or activity too low (we should potentially add 1 for current event but shoud be marginal)
            return []
        xy, t, p = event

        RC, RS = np.cos(self.Position[2]), -np.sin(self.Position[2])
        Dx, Dy = np.array(xy) - self.Position[:2]
        Dx2, Dy2 = Dx**2, Dy**2
        R2 = Dx2 + Dy2
        if R2 > self.SquaredRadius:
            return []
        self.TrackerActivity += 1
        if self.State.Disengaged and self.TrackerActivity > self.TM._TrackerDiameter * (self.TM._TrackerDisengageActivityThreshold + self.TM._TrackerDisengageActivityHysteresis):
            self.Reengage()

        self.TrackerActivity = min(self.TrackerActivity, self.Radius * 2 * self.TM._NPixelsEdge * self.TM._RelativeActivityCap)
        
        R = np.sqrt(R2)
        CurrentProjectedEvent = np.array([t, (Dx*RC - Dy*RS) / self.Position[3], (Dy*RC + Dx*RS) / self.Position[3]])
        SavedProjectedEvent = np.array(CurrentProjectedEvent) 

        if self.State.Idle:
            self.ProjectedEvents += [SavedProjectedEvent]
            if len(self.ProjectedEvents) > self.TM.TrackerMaxEventsBuffer:
                self.ProjectedEvents.pop(0)
            if self.TrackerActivity > self.TM._DetectorMinActivityForStart: 
                self.State.SetStatus(self.State._STATUS_STABILIZING)
                self.TM.StartTimes[self.ID] = t
            return [(self.Position[:2], self.ID)]

        if self.Lock:
            CurrentProjectedEvent[0] = self.Lock.Time + self.TimeConstant
            UsedEventsList = self.Lock.Events
            self.ProjectedEvents = self.ProjectedEvents[-(self.TM._TrackerLockingMaxRecoveryBuffer-1):] + [SavedProjectedEvent]
        else:
            while len(self.ProjectedEvents) > self.TM.TrackerMaxEventsBuffer or (len(self.ProjectedEvents) > 0 and self.LastUpdate - self.ProjectedEvents[0][0] >= self.EdgeBinTC): 
                self.ProjectedEvents.pop(0)
            UsedEventsList = self.ProjectedEvents
            self.ProjectedEvents += [SavedProjectedEvent]

        FlowSuccess, FlowError, ProjectionError, LocalEdge = self._ComputeLocalErrorAndEdge(CurrentProjectedEvent, UsedEventsList)
        if not FlowSuccess: # Computation could not be performed, due to not enough events
            return [(self.Position[:2], self.ID)]

        self.LastValidFlow = t
        self.FlowActivity += 1
        self.DynamicsEstimator.AddData(t, CurrentProjectedEvent[1:], FlowError, ProjectionError, self.TimeConstant*0.5)
        self.DynamicsEstimator.RecoverGeneralData()

        self.SpeedConvergenceEstimator.AddData(t, ProjectionError, self.TimeConstant)
        self.ApertureEstimator.AddData(t, LocalEdge, self.TimeConstant)

        if self.Lock:
            SpeedError = self.DynamicsEstimator.GetSpeed()
            DisplacementError = self.DynamicsEstimator.GetDisplacement()
        else:
            SpeedError = np.array([FlowError[0], FlowError[1], 0.])
            DisplacementError = np.array([ProjectionError[0], ProjectionError[1], 0.])

        SpeedMod = self.TM._NPixelsEdge * SpeedError / self.TrackerActivity
        PositionMod = self.TM._NPixelsEdge * DisplacementError / self.TrackerActivity

        self.DynamicsModifier.ModSpeed('Flow', self._CorrectModificationOrientationAndNorm(SpeedMod))
        self.DynamicsModifier.ModPosition('Flow', self._CorrectModificationOrientationAndNorm(PositionMod))

        self._Center(self)

        self._UpdateTC()

        return self.ComputeCurrentStatus(event.timestamp)

    def ComputeCurrentStatus(self, t): # Cannot be called when DEAD or IDLE. Thus, we first check evolutions for aperture issue and lock properties, then update the status
        self.State.OffCentered = (not self.State.Locked) and (np.linalg.norm(self.DynamicsEstimator.X) > self.TM._TrackerOffCenterThreshold * self.Radius)

        if not self.State.ApertureIssue and self.ApertureEstimator.Value > self.TM._TrackerApertureIssueThreshold + self.TM._TrackerApertureIssueHysteresis:
            self.State.ApertureIssue = True
            Reason = "aperture issue"
        elif self.State.ApertureIssue and self.ApertureEstimator.Value < self.TM._TrackerApertureIssueThreshold - self.TM._TrackerApertureIssueHysteresis:
            self.State.ApertureIssue = False

        CanBeLocked = True
        if self.State.ApertureIssue:
            CanBeLocked = False
        elif self.FlowActivity < (self.TM._TrackerLockedRelativeCorrectionsFailures - self.TM._TrackerLockedRelativeCorrectionsHysteresis) * self.TrackerActivity:
            CanBeLocked = False
            Reason = "unsufficient number of points matching"

        if self.State.OffCentered:
            CanBeLocked = False
            Reason = "non centered"
        elif not self.State.Locked and t < self.LastRecenter + self.TimeConstant: # We wait for the tracker to stabilize for 1px after recenter
            CanBeLocked = False
            Reason = "recent recenter"

        if self.State.Locked and not CanBeLocked:
            self.Unlock(Reason)
            self.TM._KillTracker(self, t, Reason = Reason)
            return []

        if self.State.Converged or self.State.Locked:
            if self.SpeedConvergenceEstimator.Value[0] > (self.TM._TrackerConvergenceThreshold + self.TM._TrackerConvergenceHysteresis): # Part where we downgrade to stabilizing, when the corrections are too great
                if not self.State.Locked:
                    self.State.SetStatus(self.State._STATUS_STABILIZING)
                return [(self.Position[:2], self.ID)]
            if not self.State.Locked and CanBeLocked:
                Reprojection, Spread = self.ProjectionSpread()
                if Reprojection < self.TM._LockupMaximumSpread:
                    if Spread > self.TM._LockupMaximumSpread:
                        self.TM._KillTracker(self, t, 'excessive spread')
                        return []
                    
                    self.State.SetStatus(self.State._STATUS_LOCKED)
                    self.Lock = LockClass(t, self.TrackerActivity, self.FlowActivity, list(self.ProjectedEvents + [np.array([0., 0., 0.])])) # Added event at the end allows for simplicity in neighbours search
                    
                    print("Tracker {0} has locked".format(self.ID))
        elif self.State.Stabilizing:
            if self.SpeedConvergenceEstimator.Value[0] < (self.TM._TrackerConvergenceThreshold - self.TM._TrackerConvergenceHysteresis):
                if self.FlowActivity >= (self.TM._TrackerLockedRelativeCorrectionsFailures + self.TM._TrackerLockedRelativeCorrectionsHysteresis) * self.TrackerActivity:
                    self.State.SetStatus(self.State._STATUS_CONVERGED)
        return [(self.Position[:2], self.ID)]

    def Unlock(self, Reason):
        self.Lock = None

        SpeedCancelation = -np.array(self.Speed)
        SpeedCancelation[:2] = 0
        self.DynamicsModifier.ModSpeed('Flow', SpeedCancelation)
        self.TM.FeatureManager.RemoveLock(self)
        print("Tracker {0} was released ({1})".format(self.ID, Reason))
        self.State.SetStatus(self.State._STATUS_CONVERGED)

    def ProjectionSpread(self):
        Support = np.zeros((int(2*self.Radius+1), int(2*self.Radius+1)), dtype = int)
        for Event in self.ProjectedEvents:
            Support[int(round((Event[1] + self.Radius))), int(round((Event[2] + self.Radius)))] += 1
        return ((Support==1).sum()/len(self.ProjectedEvents), (Support>0).sum() / (np.pi * self.Radius**2))

    def Disengage(self, t):
        self.State.Disengaged = True
        self.DynamicsModifier.ModPosition('Disengage', -(t - self.LastValidFlow) * self.Speed) # We go back to the last valid flow position. Nothing should have been able to change the speed between those two moments, and only the inertia modified the position
        print("Tracker {0} disengaged".format(self.ID))
    def Reengage(self): # We put that inside dedicated function for clarity.
        self.State.Disengaged = False
        print("Tracker {0} has re-engaged".format(self.ID))

    def _UpdateTC(self):
        vx, vy, w, s = self.Speed
        TrackerAverageSpeed = np.sqrt(vx**2 + vy**2 + self.DynamicsEstimator.r2 * (w**2 + s**2) + 2 * (self.DynamicsEstimator.X * np.array([vy*w + vx*s, -vx*w + vy*s])).sum()) # If dynamics estimator is computed from tracker center
        if TrackerAverageSpeed == 0:
            self.TimeConstant = self.TM._TrackerDefaultTimeConstant
        else:
            self.TimeConstant = min(self.TM._TrackerDefaultTimeConstant, self.TM._NPixelsEdge / TrackerAverageSpeed)
        self.EdgeBinTC = self.TimeConstant * self.TM._EdgeBinRatio

    def _CorrectModificationOrientationAndNorm(self, Mod):
        cs, ss = self.Position[3]*np.cos(self.Position[2]), self.Position[3]*np.sin(self.Position[2])
        M = np.array([[cs, -ss], [ss, cs]])
        Mod[:2] = M.dot(Mod[:2])
        return Mod

    def _Center(self):
        if self.State.Converged and self.State.OffCentered and self.LastUpdate > self.LastRecenter + self.TimeConstant*2:
            Shift = np.array([0., 0., 0., 0.])
            Shift[:2] = self.TM._TrackerMeanPositionDisplacementFactor * np.array(self.DynamicsEstimator.X)
            self.DynamicsEstimator.TrackerShift(Shift[:2])

            ProjectedEvents = []
            for Event in self.ProjectedEvents:
                NewLoc = Event[1:] - Shift[:2]
                if np.linalg.norm(NewLoc) <= self.Radius:
                    ProjectedEvents += [np.array([Event[0], NewLoc[0], NewLoc[1]])]
            self.ProjectedEvents = ProjectedEvents

            self.DynamicsModifier.ModPosition('MeanPos', Shift)
            self.LastRecenter = self.LastUpdate

    def _ComputeLocalErrorAndEdge(self, CurrentProjectedEvent, PreviousEvents):
        ConsideredNeighbours = []
        LocalEdgePoints = []
        LocalEdgeSquaredPoints = []
        for PreviousEvent in reversed(PreviousEvents[:-1]): # We reject the last event, that is either the CPE, or a fake event in Lock.Events
            D = np.linalg.norm(CurrentProjectedEvent[1:] - PreviousEvent[1:])
            if D < self.TM.LocalEdgeRadius:
                if len(LocalEdgePoints) < self.TM.LocalEdgeNumberOfEvents:
                    LocalEdgePoints += [PreviousEvent[1:]]
                if D < self.TM._ClosestEventProximity:
                    ConsideredNeighbours += [np.array(PreviousEvent)]
                    if len(ConsideredNeighbours) == self.TM._MaxConsideredNeighbours:
                        break

        if len(ConsideredNeighbours) < self.TM.MinConsideredNeighbours:
            return False, np.array([0., 0.]), np.array([0., 0.]), np.array([0., 0.])

        SpeedError, DeltaPos, MeanPoint = self._ComputeSpeedError(self, CurrentProjectedEvent, np.array(ConsideredNeighbours))

        LocalEdgePoints = np.array(LocalEdgePoints)
        LocalEdgeVectors = np.zeros(LocalEdgePoints.shape)
        LocalEdgeVectors[:,0] = LocalEdgePoints[:,0] - MeanPoint[0]
        LocalEdgeVectors[:,1] = LocalEdgePoints[:,1] - MeanPoint[1]
        LocalEdgeNorms = np.linalg.norm(LocalEdgeVectors, axis = 1)

        LocalEdgeRotatedVectors = RotateVectors(LocalEdgeVectors)
        LocalEdge = LocalEdgeRotatedVectors.mean(axis = 0)
        NEdge = np.linalg.norm(LocalEdge)
        if NEdge > 0:
            LocalEdge /= NEdge

        xy = (1+np.array([LocalEdge[0], -LocalEdge[0]]))/2
        deltaxy = np.sqrt(abs(xy))
        if LocalEdge[1] < 0:
            deltaxy[0] *= -1
        SpeedError = SpeedError - ((SpeedError*deltaxy).sum()) * deltaxy
        DeltaPos = DeltaPos - ((DeltaPos*deltaxy).sum()) * deltaxy

        return True, SpeedError, DeltaPos, LocalEdge

    def _ComputeSpeedError(self, CurrentProjectedEvent, ConsideredNeighbours):
        MeanPoint = ConsideredNeighbours.mean(axis = 0)

        DeltaPos = (CurrentProjectedEvent - MeanPoint)
        SpeedError = DeltaPos[1:] / DeltaPos[0]
        return SpeedError, DeltaPos[1:], MeanPoint[1:]

