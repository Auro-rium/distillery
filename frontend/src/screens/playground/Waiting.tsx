import { useEffect, useState } from "react";
import { Card } from "../../components";
import { Spinner } from "../../ui";

/**
 * Shown while a question is in flight. The server does not stream, so this only says what it is doing in general and
 * how long the visitor has waited (a clock in the browser, not a server figure).
 */
export function Waiting() {
  const [s, setS] = useState(0);
  useEffect(() => {
    const t0 = Date.now();
    const id = setInterval(() => setS(Math.floor((Date.now() - t0) / 1000)), 1000);
    return () => clearInterval(id);
  }, []);
  return (
    <Card className="pg-wait">
      <Spinner inline label="Asking the Acme analyst" />
      <p className="pg-wait-clock" aria-hidden="true"><span className="num">{s}</span> s</p>
      <ol className="pg-wait-steps">
        <li>The question is compared with what the student was certified on. If it looks unfamiliar, the teacher answers in a few seconds.</li>
        <li>Otherwise the student writes the SQL on a Nebius CPU sandbox and the SQL is run read-only. This is the slow part: usually about a minute, longer for the first question after a quiet spell while the sandbox starts.</li>
        <li>The router scores the student's answer. Below the certified threshold, the teacher answers instead and the card says why.</li>
      </ol>
    </Card>
  );
}
