// How it works (`/how-it-works`): a guided tour of the method in two acts (build the student, serve it safely),
// then the whole method as one diagram. It carries no numbers and no run data: the tour's words are in steps.ts,
// the diagram and its text list are the replay screen's Pipeline.
import { Pipeline } from "../replay/Pipeline";
import { BUILD, SERVE } from "./steps";
import { Tour } from "./Tour";
import "./howitworks.css";

const PROMISES: { title: string; body: string }[] = [
  { title: "Run, not trusted", body: "Every answer and every lesson is executed before it counts." },
  { title: "Sealed exam", body: "The final test is locked away and opened exactly once." },
  { title: "No AI judge", body: "PROMOTE or REJECT is decided by plain code with fixed rules." },
  { title: "Unsure? Teacher.", body: "The small model only answers where its accuracy is proven." },
];

export default function HowItWorks() {
  return (
    <div className="stack how">
      <section className="hero how-hero" aria-labelledby="how-h">
        <h1 id="how-h">How it works</h1>
        <p className="hero-copy muted">
          A big model teaches a small one a single job, every step is checked by running the SQL, and the small model
          only answers where it has proven itself. Take the tour.
        </p>
      </section>

      <ul className="how-promises" role="list" aria-label="What the method promises">
        {PROMISES.map((p) => (
          <li key={p.title} className="how-promise">
            <strong>{p.title}</strong>
            <span className="muted">{p.body}</span>
          </li>
        ))}
      </ul>

      <Tour title="Act one: build the student" lead="From a pile of questions to a small model that passed a sealed exam." steps={BUILD} />
      <Tour title="Act two: serve it safely" lead="How the small model is put to work without risking wrong answers." steps={SERVE} />

      <section className="how-whole" aria-labelledby="how-whole-h">
        <h2 id="how-whole-h">The whole picture</h2>
        <Pipeline />
      </section>
    </div>
  );
}
