/**
 * THE SCRIPT — edit this file to change the video.
 *
 * Each scene = a visual (`scene`, see src/scenes/index.ts) + narration `beats`.
 * One beat = one caption line = one voice clip. Visuals key their animations to beat
 * numbers (0-based), so rewording a line re-times everything automatically.
 *
 * Caption markup:
 *   *word*                 -> highlighted in the caption (spoken normally)
 *   {shown|spoken}         -> caption shows "shown", the voice says "spoken"
 *   10^120                 -> superscript in the caption
 *
 * After editing narration: `npm run voice` (regenerates only changed clips), then
 * `npm run studio` to preview or `npm run render` for the MP4.
 */
import type {SceneDef} from './types';

export const VIDEO_TITLE = 'What If Dark Energy Is Just… the Neighbours?';
export const VIDEO_SUBTITLE = 'The Progenitor Node Model, explained';

export const CREDITS = {
  paper:
    'Replacing Dark Energy and Unifying Large-Scale Anomalies: The Progenitor Node Model within a Virialized Meta-Structure',
  author: 'Martin Gamsby',
  orcid: 'ORCID 0009-0007-4069-9687',
  code: 'github.com/MartinGamsby/ProgenitorNodeModelCosmology',
  doi: 'doi.org/10.5281/zenodo.20481859',
};

export const SCENES: SceneDef[] = [
  {
    id: 'hook',
    scene: 'Hook',
    chapter: '',
    beats: [
      'Everything in the universe is flying apart.',
      "And here's the weird part: it's *speeding up*.",
      'Physicists call whatever is doing this *dark energy*.',
      "It's about seventy percent of everything… and nobody knows what it is.",
      "So here's a what-if.",
      'What if nothing is *pushing* the universe apart?',
      'What if something is *pulling*?',
    ],
  },
  {
    id: 'title',
    scene: 'Title',
    chapter: '',
    beats: [''],
    minSeconds: 4.5,
    sfx: [{beat: 0, sound: 'boom'}],
  },
  {
    id: 'problem',
    scene: 'Problem',
    chapter: 'The problem',
    beats: [
      'In 1998, two teams measured distant exploding stars: {Type Ia|type one A} supernovae.',
      'They were too dim. Farther away than a slowing-down universe allows.',
      'The fix: a cosmological constant, {Λ|Lambda}. Energy built into empty space itself.',
      'It works beautifully. But it comes with two embarrassing problems.',
      'One: quantum physics predicts a vacuum energy about {10^120|ten to the one hundred and twenty} times too big.',
      'Two: why is dark energy roughly equal to matter *right now*, of all the moments we could be living in?',
      'Fitting the data is not the same as *explaining* it.',
    ],
  },
  {
    id: 'beyond',
    scene: 'Beyond',
    chapter: 'Beyond the horizon',
    beats: [
      "Let's zoom out.",
      'Everything we can ever see is a sphere about 14 {gigaparsecs|giga-parsecs} in radius. Roughly 46 billion light-years.',
      'Standard cosmology assumes it looks the same out there as in here. Forever.',
      'But what if it doesn’t?',
      'Picture giant dark masses far beyond our horizon, each *hundreds to thousands* of times heavier than our entire observable universe.',
      'Call them {HMEAs|H. M. E. A. s}: *hyper-massive external attractors*.',
      'Not a neat grid: a lumpy, settled swarm, like galaxies in a relaxed cluster. Only absurdly bigger.',
      'And our universe sits in one of the gaps.',
    ],
  },
  {
    id: 'tides',
    scene: 'Tides',
    chapter: 'Tides',
    beats: [
      'Quick gravity fact: a perfectly smooth shell of mass around you pulls on you… *not at all*.',
      "Every direction cancels. That's Newton's shell theorem.",
      'But *lumps* don’t cancel.',
      'The side of our bubble closer to a node gets pulled harder than the middle, and the middle harder than the far side.',
      "So the bubble gets *stretched*. That's a tide, the same thing the Moon does to our oceans.",
      'Do the math, and the tidal pull grows in proportion to distance: {2GMR/S³|two G M R over S cubed}.',
      "And dark energy's push? Also proportional to distance.",
      'Same shape. To gravity, a *pull from outside* can look exactly like a *push from inside*.',
    ],
  },
  {
    id: 'numbers',
    scene: 'Numbers',
    chapter: 'How big? How far?',
    beats: [
      'How heavy, and how far? Plug in the numbers.',
      "Nodes about five hundred times our observable universe's mass…",
      '…spaced about 39 {gigaparsecs|giga-parsecs} apart, would do the job.',
      'No exotic vacuum energy. Just ordinary gravity, from very big things, very far away.',
    ],
  },
  {
    id: 'whatif-center',
    scene: 'WhatIfCenter',
    chapter: 'What if…',
    whatIf: 'we sat dead centre?',
    beats: [
      'What if we sat exactly in the middle, with nodes evenly all around us?',
      'Then every pull cancels. No acceleration at all.',
      'And a subtle twist: in empty space, tides can’t simply inflate a volume.',
      'They stretch one way and squeeze the others. Pure *shear*.',
      'The overall speed-up comes from the bubble creeping closer to the nodes, where the pull grows faster and faster.',
      'And that leftover stretch? It’s a *fingerprint*. Hold that thought.',
    ],
  },
  {
    id: 'progenitor',
    scene: 'Progenitor',
    chapter: 'The Progenitor Hypothesis',
    beats: [
      'But wait. Why would we be anywhere near the middle of a gap?',
      "Here's the boldest what-if: what if the Big Bang *was one of these nodes*?",
      'A node, sitting at its balance point among its neighbours…',
      '…destabilises.',
      'It blows apart, and the matter expands evenly into the surrounding gap.',
      'That explains why we start near the middle, and why the expansion looks the same in every direction.',
      "This is the *Progenitor Hypothesis*. And it's one mass playing two roles:",
      'before the bang, a member of the swarm. After it, everything that came out.',
      'The most natural guess: our progenitor was just a *typical* node. Nothing special.',
    ],
    sfx: [{beat: 3, sound: 'boom'}],
  },
  {
    id: 'simulation',
    scene: 'Simulation',
    chapter: 'The simulation',
    beats: [
      'Nice story. Does it actually work? Time to simulate.',
      'Thousands of particles in a cloud, and nothing but plain Newtonian gravity.',
      'Self-gravity slows the cloud down. The distant nodes pull it apart.',
      'We start 2.9 billion years after the Big Bang, and run to today: 13.8 billion years.',
      'First, a sanity check: set the node mass to *zero*.',
      'The simulation hands back the textbook universe with no dark energy. Exactly.',
      'Good. Now turn the nodes *on*.',
      'The cloud grows by just the right amount. Like a universe with dark energy, but without any.',
    ],
  },
  {
    id: 'data',
    scene: 'Data',
    chapter: 'Moment of truth',
    beats: [
      'Now the moment of truth: *real data*.',
      'Over fifteen hundred {Type Ia|type one A} supernovae from the {Pantheon+|Pantheon plus} catalogue.',
      'We turn the simulated expansion into a distance-versus-redshift curve. No shortcuts.',
      'Zoom in on the differences. A universe without dark energy misses. *Badly*.',
      'Standard {ΛCDM|Lambda C D M} nails it.',
      'And the node model?',
      'Right on top of it.',
      'Fit score, lower is better: no dark energy, {0.843|zero point eight four three}. {ΛCDM|Lambda C D M}, {0.436|zero point four three six}. Node model, {0.440|zero point four four zero}.',
      'To be clear: it *matches* {ΛCDM|Lambda C D M}. It doesn’t beat it. And the match holds from ten thousand to a hundred thousand particles.',
    ],
    sfx: [{beat: 6, sound: 'ding'}],
  },
  {
    id: 'whatif-observer',
    scene: 'WhatIfObserver',
    chapter: 'What if…',
    whatIf: "we're not at the centre?",
    beats: [
      "What if we're not at the centre of the cloud? We almost certainly aren't.",
      "So every particle gets a turn being *us*. Does any vantage point see our universe?",
      'Plenty do. And not just in one lucky setup: *many* different configurations get there.',
    ],
  },
  {
    id: 'whatif-close',
    scene: 'WhatIfClose',
    chapter: 'What if…',
    whatIf: 'the nodes are too close?',
    beats: [
      'What if the nodes are too close? Two things can go wrong.',
      'The pull wins outright: a *runaway* that rips the cloud apart way too fast.',
      'Or something sneakier: the core of the cloud collapses into a dense *knot*.',
      'On the left, a run with a great supernova fit… and a collapsing core. On the right, a healthy one.',
      'So a good fit is not enough. A universe-like run must pass a checklist:',
      'fit the supernovae, grow by the right amount, keep its core from collapsing, and keep every node outside our horizon.',
      'Picking by fit alone would have chosen the collapsing universes. The checklist catches them.',
      'With it: *68 out of 72* configurations are universe-like.',
    ],
  },
  {
    id: 'whatif-mass',
    scene: 'WhatIfMass',
    chapter: 'What if…',
    whatIf: 'the progenitor was tiny? Or huge?',
    beats: [
      'What if the progenitor was tiny? Or enormous?',
      'We tried more than three orders of magnitude: from one observable universe, up to three thousand.',
      'The fit barely cares. The structure does.',
      'And the most robust choice is the *typical node*: exactly what the hypothesis preferred, before we ran anything.',
    ],
  },
  {
    id: 'catch',
    scene: 'Catch',
    chapter: 'The catch',
    beats: [
      'Now the catch. Because there is one.',
      'Heavy nodes far away, or lighter nodes closer in: the supernovae can’t tell them apart.',
      'The average expansion only measures mass over spacing cubed.',
      'So the Hubble diagram alone will never prove this model. We need a different test.',
    ],
  },
  {
    id: 'predictions',
    scene: 'Predictions',
    chapter: 'Predictions',
    beats: [
      'Remember the stretch fingerprint? *That’s* the test.',
      'Prediction one: a *Hubble dipole*. One half of the sky should expand a little faster than the other.',
      'Which might feed the Hubble tension: 67 versus 73, depending on how you measure.',
      'Prediction two: the *Axis of Evil*. The biggest patterns in the cosmic microwave background line up along a strange axis.',
      'Lopsided neighbours could imprint exactly that kind of alignment.',
      'Prediction three: *dark flow*. Galaxies drifting in bulk, pulled by something outside the survey.',
      'Prediction four: as we near the nodes, the acceleration *grows*. Phantom dark energy: {w < −1|w below minus one}.',
      'And the kicker: they should all point *the same way*. Toward our nearest, heaviest neighbours.',
    ],
  },
  {
    id: 'whatif-invisible',
    scene: 'WhatIfInvisible',
    chapter: 'What if…',
    whatIf: 'they’re real? Why can’t we see them?',
    beats: [
      'If these things are so huge, why can’t we see them?',
      "They're beyond our horizon, for a start. But they'd also be ancient black holes that ate everything nearby long ago.",
      'No food, no glowing disk. Their Hawking temperature: about {10^-33|ten to the minus thirty-three} kelvin.',
      '*Dark giants*. Seen only by their pull.',
    ],
  },
  {
    id: 'whatif-cycle',
    scene: 'WhatIfCycle',
    chapter: 'What if…',
    whatIf: 'it’s a cycle?',
    beats: [
      'One last, wild what-if. Pure speculation: no simulation behind this one.',
      'A node bangs. Its matter fills the gap.',
      'Over eons, the neighbours swallow that matter.',
      'One grows too big, destabilises… and bangs again.',
      'Our universe would be one breath in a *great cosmic metabolism*.',
    ],
  },
  {
    id: 'limits',
    scene: 'Limits',
    chapter: 'Honesty corner',
    beats: [
      "Let's be honest about what this is: a *toy model*.",
      'Newtonian gravity in flat space. No general relativity, no early universe, no microwave background spectrum, no baryon acoustic oscillations.',
      "It doesn't claim this is reality. It claims ordinary gravity can fake dark energy. And that's worth checking.",
    ],
  },
  {
    id: 'outro',
    scene: 'Outro',
    chapter: '',
    beats: [
      'So. If the Hubble dipole, the Axis of Evil, and the dark flow all line up…',
      '…then dark energy might not be a property of space at all.',
      'It might just be *the neighbours*.',
      '',
    ],
    minSeconds: 18,
  },
];
