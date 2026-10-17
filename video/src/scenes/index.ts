import React from 'react';
import {Beyond} from './Beyond';
import {Data} from './Data';
import {Catch, Limits, Outro, WhatIfCycle, WhatIfInvisible} from './Ending';
import {Predictions} from './Predictions';
import {Hook, TitleCard} from './Intro';
import {Numbers} from './cut/Numbers';
import {TidesOriginal} from './cut/TidesOriginal';
import {WhatIfCenterOriginal} from './cut/WhatIfCenterOriginal';
import {Problem} from './Problem';
import {Progenitor} from './Progenitor';
import {Simulation} from './Simulation';
import {Teaser} from './cut/Teaser';
import {Tides} from './Tides';
import {WhatIfCenter} from './WhatIfCenter';
import {WhatIfClose, WhatIfMass, WhatIfObserver} from './WhatIfStructure';

/** Scene key (the `scene` field in script.ts) -> component. */
export const SCENE_COMPONENTS: Record<string, React.FC> = {
  Hook,
  Title: TitleCard,
  Problem,
  Beyond,
  Tides,
  WhatIfCenter,
  Progenitor,
  Simulation,
  Data,
  WhatIfObserver,
  WhatIfClose,
  WhatIfMass,
  Catch,
  Predictions,
  WhatIfInvisible,
  WhatIfCycle,
  Limits,
  Outro,
  // cutting room (src/cutting-room.ts): not in the video, restorable
  Numbers,
  Teaser,
  TidesOriginal,
  WhatIfCenterOriginal,
};
