/**
 * THE SCRIPT — edit this file to change the video.
 *
 * Each scene = a visual (`scene`, see src/scenes/index.ts) + narration `beats`.
 * One beat = one caption line = one voice clip. Visuals key their animations to beat
 * numbers (0-based), so rewording a line re-times everything automatically.
 * Removed scenes live in src/cutting-room.ts (kept for future videos).
 *
 * Caption markup:
 *   *word*                 -> highlighted in the caption (spoken normally)
 *   {shown|spoken}         -> caption shows "shown", the voice says "spoken"
 *   10^120                 -> superscript in the caption
 *
 * Writing for the voice: short sentences, commas and periods for pauses (no "…"), a question
 * mark when it's a question, spoken forms for symbols and numbers via {shown|spoken}.
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
  code: 'github.com/MartinGamsby/ProgenitorNodeModelCosmology',
  doi: 'doi.org/10.5281/zenodo.20481859',
};

// 'Lambda CDM' (CDM as one word) gave the shortest Lambda→CDM gap and the cleanest transcript
const LCDM = '{ΛCDM|Lambda CDM}';

export const SCENES: SceneDef[] = [
  {
    id: 'hook',
    stars: false,
    scene: 'Hook',
    // silent cold open before the first line: swarm → a bang → neighbours feed → one swells,
    // and it bangs on "flying apart" (see scenes/Intro.tsx)
    leadSeconds: 8.5,
    voice: {1: {exaggeration: 0.55}}, // "It's speeding up" read calmer
    chapter: '',
    beats: [
      'Everything in the universe is flying apart.',
      "And here's the weird part. It's *speeding up*.",
      'Physicists call whatever is doing this: *dark energy*.',
      "It makes up about seventy percent of everything. And nobody knows what it is.",
      "So here's a what-if.",
      'What if nothing is *pushing* the universe apart?',
      'What if something is *pulling* it?',
    ],
  },
  {
    id: 'title',
    stars: false,
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
      'In 1998, two teams measured distant exploding stars, called {Type Ia supernovae|type 1A supernovas}.',
      'They looked too dim. They were farther away than a slowing universe would allow.',
      'The fix was a cosmological constant, {Λ|Lambda}: energy built into empty space itself.',
      `Add invisible dark matter, and you get today's standard model of the universe, called ${LCDM}. Lambda for dark energy, CDM for cold dark matter.`,
      'It works beautifully. But it comes with two embarrassing problems.',
      'First, quantum physics predicts a vacuum energy too big by a factor of {10^120: a one|one} followed by {120|a hundred and twenty} zeros.',
      'Second, why is dark energy about equal to matter *right now*, of all the moments we could be living in?',
      'Fitting the data is not the same as *explaining* it.',
    ],
  },
  {
    id: 'beyond',
    stars: false,
    scene: 'Beyond',
    chapter: 'Beyond the horizon',
    beats: [
      "Let's zoom out.",
      'Everything we can ever see fits in a sphere about forty-six billion light-years in radius.',
      "Standard cosmology assumes that beyond our horizon, there's just more of the same: galaxies like ours, in every direction, forever.",
      "But what if it only goes so far? Out to the edge of what the Big Bang threw out, and beyond that, empty space.",
      'Picture giant dark masses far beyond our horizon, each *hundreds to thousands* of times heavier than our entire observable universe.',
      'Call them *hyper-massive external attractors*.',
      "They don't form a neat grid. They form a lumpy, settled swarm, like galaxies in a relaxed cluster, only absurdly bigger.",
      "Physicists call that *virialized*. Not a frozen crystal where everything sits still, but a swarm where everything keeps moving, and the whole thing keeps its shape.",
      'And our universe sits in one of the gaps.',
    ],
  },
  {
    id: 'tides',
    stars: false,
    scene: 'Tides',
    chapter: 'Pulled apart',
    beats: [
      "Here's a gravity fact that sounds wrong. Surround yourself with a perfectly smooth shell of mass, and it won't pull you *at all*.",
      "Every direction cancels out. That's Newton's shell theorem.",
      "But *clumps* don't cancel. The side of our universe facing a clump gets pulled harder than the far side, so it gets pulled apart.",
      'And real clumps come in *different sizes*, at *different distances*. That makes the pull even more lopsided.',
      'An outward pull that gets stronger the farther out you go. That is exactly what dark energy seems to do.',
      'So to gravity, a *pull from outside* can look just like a *push from inside*. No exotic energy required.',
    ],
  },
  {
    id: 'whatif-center',
    stars: false,
    scene: 'WhatIfCenter',
    chapter: 'What if…',
    whatIf: 'we sat dead centre?',
    beats: [
      'What if we sat exactly in the middle, with the neighbours spread evenly all around us?',
      'Then every pull would cancel out, and nothing would speed up.',
      'But a real swarm is never perfectly even. One side always pulls a little harder.',
      "So our universe should expand a little faster in one direction. Keep that in mind. It's a *fingerprint* we can actually look for.",
    ],
  },
  {
    id: 'progenitor',
    stars: false,
    scene: 'Progenitor',
    chapter: 'The Progenitor Hypothesis',
    beats: [
      'But wait. Why would we be anywhere near the middle of a gap?',
      'What if the Big Bang *was one of these attractors*?',
      'Picture one of them, sitting at its balance point among its neighbours.',
      'Then it {destabilises|becomes unstable}.',
      'It blows apart, and its matter spreads out evenly into the surrounding gap.',
      'That explains why we start near the middle, and why the expansion looks the same in every direction.',
      "This is the *Progenitor Hypothesis*. And it means one mass plays two roles.",
      'Before the bang, it was one member of the swarm. After the bang, that same mass became our whole universe, plus whatever lies beyond what we can see.',
      'The most natural guess is that our progenitor was just a *typical* member. Nothing special.',
    ],
    sfx: [{beat: 3, sound: 'boom'}],
  },
  {
    id: 'simulation',
    stars: false,
    scene: 'Simulation',
    chapter: 'The simulation',
    beats: [
      'Nice story. But does it actually work? Time to simulate.',
      'We take thousands of particles in a cloud, and nothing but plain Newtonian gravity.',
      "The cloud's own gravity slows it down. The distant attractors pull it apart.",
      'We start {2.9|two point nine} billion years after the Big Bang, and run all the way to today, {13.8|thirteen point eight} billion years.',
      'First, a sanity check. We set the attractor mass to *zero*.',
      'The simulation gives back the textbook universe with no dark energy. Exactly.',
      'Good. Now we turn the attractors *on*.',
      'The cloud grows by just the right amount, like a universe with dark energy. Without any.',
    ],
  },
  {
    id: 'data',
    scene: 'Data',
    chapter: 'Moment of truth',
    beats: [
      'Now, the moment of truth: *real data*.',
      'More than fifteen hundred {Type Ia supernovae|type 1A supernovas}, from the {Pantheon+|Pantheon plus} catalogue.',
      `This is the same kind of data that gave ${LCDM} its numbers in the first place. Now we hold our simulation up to it, the exact same way.`,
      'We turn the simulated expansion into a curve of distance versus redshift, straight from the particles.',
      'Now zoom in on the differences. A universe without dark energy misses. *Badly*.',
      `Standard ${LCDM} nails it.`,
      'And the attractor model?',
      'Right on top of it.',
      `Here's the fit score, where lower is better. No dark energy: {0.843|zero point eight four three}. ${LCDM}: {0.436|zero point four three six}. Our model: {0.440|zero point four four zero}.`,
      `Some runs even score {0.433|zero point four three three}, a hair better than ${LCDM}. But that gap is smaller than the simulation's own noise, so the honest word is: it *matches*.`,
      'And with enough tuning, some other setup would surely score even better. But that would prove nothing.',
      "As a software developer, this is something I watch for all the time. It's called *overfitting*. Give a model enough knobs, and it will fit almost anything. Simulations can fall into the same trap.",
      'Fitting is not explaining. So a good fit is only the first test. Next, we check whether the simulated universe actually looks like ours.',
    ],
  },
  {
    id: 'whatif-observer',
    stars: false,
    scene: 'WhatIfObserver',
    chapter: 'What if…',
    whatIf: "we're not at the centre?",
    beats: [
      "What if we're not at the centre of the cloud? We almost certainly aren't.",
      "So every particle gets a turn being *us*. Does any vantage point see a universe like ours?",
      `Plenty do. And not in just one lucky setup. Across seventy-one configurations, the best vantage lands right around ${LCDM}.`,
    ],
  },
  {
    id: 'whatif-close',
    stars: false,
    scene: 'WhatIfClose',
    chapter: 'What if…',
    whatIf: 'the attractors are too close?',
    beats: [
      'What if the attractors are too close? Two things can go wrong.',
      'The pull can win outright, in a *runaway* that rips the cloud apart way too fast.',
      'Or something sneakier happens. The core of the cloud collapses into a dense *knot*.',
      "On the left, a run with a decent supernova fit, and a collapsing core. On the right, a healthy one.",
      "So a good fit isn't enough. A universe-like run has to pass a checklist.",
      'It must fit the {supernovae|supernovas}, grow by the right amount, keep its core from collapsing, and keep every attractor outside our horizon.',
      'Picking runs by fit alone would have chosen the collapsing universes. The checklist catches them.',
      'We ran thousands of simulations to map where universes like ours can form. Inside the best region, we re-ran 72 configurations with more particles, and *68 of them* pass the whole checklist.',
    ],
  },
  {
    id: 'whatif-mass',
    scene: 'WhatIfMass',
    chapter: 'What if…',
    whatIf: 'the progenitor was tiny? Or huge?',
    beats: [
      'What if the progenitor was tiny? Or enormous?',
      'We tried more than three orders of magnitude, from one observable universe up to three thousand.',
      'The fit barely cares. The structure does.',
      'And the most robust choice is the *typical* member, exactly what the hypothesis preferred before we ran anything.',
    ],
  },
  {
    id: 'catch',
    stars: false,
    scene: 'Catch',
    chapter: 'The catch',
    beats: [
      "Now, the catch. Because there is one.",
      "Heavy attractors far away, or lighter ones closer in. The {supernovae|supernovas} can't tell them apart.",
      'The average expansion only measures the mass divided by the spacing cubed.',
      'So the {supernovae|supernovas} alone will never prove this model. We need a different test.',
    ],
  },
  {
    id: 'predictions',
    scene: 'Predictions',
    chapter: 'Predictions',
    beats: [
      "Remember that lopsided fingerprint? *That's* the test.",
      'Prediction one: a *Hubble dipole*. One half of the sky should expand a little faster than the other.',
      'Measure the expansion rate from the whole sky at once, and you get 67. Measure it with nearby galaxies, and you get 73.',
      "Those numbers should agree, and they don't. It's called the *Hubble tension*.",
      'For standard cosmology, that is a headache. For this model, it is expected. A lopsided expansion, what physicists call *anisotropy*, gives different answers depending on how you look.',
      "Prediction two: the *Axis of Evil*. The biggest patterns in the cosmic microwave background all point along the same strange axis.",
      'Lopsided neighbours could imprint exactly that kind of alignment.',
      "Prediction three: *dark flow*. Huge regions of galaxies drifting together, as if pulled by something we can't see.",
      'Prediction four: the acceleration keeps growing. As we drift toward the attractors, their pull gets stronger, so the speed-up itself speeds up.',
      "Cosmologists call that *phantom* dark energy. A cosmological constant can't do it, because its push never changes.",
      'And the kicker: all four should point *the same way*, toward our nearest, heaviest neighbours.',
    ],
  },
  {
    id: 'whatif-invisible',
    stars: false,
    scene: 'WhatIfInvisible',
    chapter: 'What if…',
    whatIf: 'they’re real? Why can’t we see them?',
    beats: [
      "If these things are so massive, why can't we see them?",
      "Distance isn't the answer. They're older than the Big Bang, so their light would have had plenty of time to reach us.",
      "The answer is that they have no light left to give. They'd be ancient black holes that swallowed everything around them long ago, and a black hole with nothing left to eat doesn't glow.",
      "Even their Hawking radiation is absurdly cold: about {10^-33|ten to the minus thirty-three} kelvin. That's as close to absolute zero as it gets.",
      '*Dark giants*. Seen only by their pull.',
    ],
  },
  {
    id: 'whatif-cycle',
    stars: false,
    scene: 'WhatIfCycle',
    chapter: 'What if…',
    whatIf: 'it’s a cycle?',
    beats: [
      'One last, wild what-if. This one is pure speculation, with no simulation behind it.',
      'An attractor bangs, and its matter fills the gap.',
      'Over eons, the neighbours swallow that matter.',
      'One of them grows too big, becomes unstable, and bangs again.',
      'Our universe would be one breath in a *great cosmic metabolism*.',
    ],
  },
  {
    id: 'limits',
    scene: 'Limits',
    chapter: 'Honesty corner',
    beats: [
      "Let's be honest about what this is. It's a *toy model*.",
      'Newtonian gravity in flat space. No general relativity, no early universe, no microwave background spectrum, and no baryon acoustic oscillations.',
      "It doesn't claim to be reality. It claims that ordinary gravity can fake dark energy. And that's worth checking.",
    ],
  },
  {
    id: 'outro',
    stars: false,
    scene: 'Outro',
    chapter: '',
    beats: [
      'So, if the Hubble dipole, the Axis of Evil, and the dark flow all line up,',
      'then dark energy might not be a property of space at all.',
      'It might just be *the neighbours*.',
      '',
    ],
    minSeconds: 20,
  },
];
