# eb-tracker
Asynchronous Event-Based Tracker

Standalone version of the code for the event-based tracking algorithm : "An Event-by-Event Feature Detection and Tracking Invariant to Motion Direction and Velocity".
This was initially a module for a yet unpublished work-in-progress event-based framework. Thus, we publish this standalone version.
It has been cleared of unused parameters, unstable features, and data post-processing.

This version has not been thoroughly tested, thus issues may remain. However, the core algorithm remains valid.
Instanciate a class TrackerModule, then run each event tuple ((x,y), t, p) through the OnEvent class method.
Output should be a list of tuples ((x_T, y_T), ID_T) for each locked tracker affected by this event. (x_T, y_T) defines the tracker location on screen, and ID_T is its identifier.

Please report any bug on the dedicated git repository, for us to fix it as soon as possible.
