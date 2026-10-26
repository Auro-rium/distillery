// How it works (`/how-it-works`): the method as one picture. It carries no numbers and no run data; the diagram and its
// text list live in the replay screen's Pipeline, which the Mission page used to embed.
import { Pipeline } from "../replay/Pipeline";

export default function HowItWorks() {
  return (
    <div className="stack how">
      <section className="hero" aria-labelledby="how-h">
        <h1 id="how-h">How it works</h1>
        <p className="hero-copy muted">Every example and every evaluation is checked by running the SQL.</p>
      </section>
      <Pipeline />
    </div>
  );
}
