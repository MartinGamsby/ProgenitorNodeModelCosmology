import React from 'react';
import {Beyond} from './Beyond';
import {Data} from './Data';
import {Catch, Limits, Outro, Predictions, WhatIfCycle, WhatIfInvisible} from './Ending';
import {Hook, TitleCard} from './Intro';
import {Numbers} from './Numbers';
import {Problem} from './Problem';
import {Progenitor} from './Progenitor';
import {Simulation} from './Simulation';
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
  Numbers,
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
};
