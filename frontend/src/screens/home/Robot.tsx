import "./robot.css";

/**
 * A decorative plush robot that dances beside the hero. Pure illustration: no data, hidden from assistive
 * technology. The dance runs only when the user has not asked for reduced motion (see robot.css).
 */
export function Robot() {
  return (
    <div className="h-robot" aria-hidden="true">
      <svg viewBox="0 0 320 390" focusable="false">
        <defs>
          <linearGradient id="rb-plush" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#fbfefd" />
            <stop offset=".55" stopColor="#dce8e5" />
            <stop offset="1" stopColor="#b7cbc6" />
          </linearGradient>
          <linearGradient id="rb-limb" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#eef6f4" />
            <stop offset="1" stopColor="#bdd0cb" />
          </linearGradient>
          <linearGradient id="rb-visor" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#14303a" />
            <stop offset="1" stopColor="#07141a" />
          </linearGradient>
          <linearGradient id="rb-belly" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#1c5a45" />
            <stop offset="1" stopColor="#0d2f26" />
          </linearGradient>
          <radialGradient id="rb-shine" cx=".5" cy=".5" r=".5">
            <stop offset="0" stopColor="#fff" stopOpacity=".75" />
            <stop offset="1" stopColor="#fff" stopOpacity="0" />
          </radialGradient>
          <filter id="rb-glow" x="-60%" y="-60%" width="220%" height="220%">
            <feGaussianBlur stdDeviation="4" />
          </filter>
        </defs>

        <ellipse className="rb-shadow" cx="160" cy="366" rx="88" ry="11" />

        <g className="rb-sway">
          <g className="rb-bob">
            {/* legs and feet */}
            <g className="rb-leg rb-leg-l">
              <rect x="108" y="296" width="48" height="54" rx="24" fill="url(#rb-limb)" />
              <ellipse cx="132" cy="350" rx="33" ry="15" fill="url(#rb-plush)" />
              <path d="M104 350 q28 11 56 0" className="rb-stitch" />
            </g>
            <g className="rb-leg rb-leg-r">
              <rect x="164" y="296" width="48" height="54" rx="24" fill="url(#rb-limb)" />
              <ellipse cx="188" cy="350" rx="33" ry="15" fill="url(#rb-plush)" />
              <path d="M160 350 q28 11 56 0" className="rb-stitch" />
            </g>

            {/* arms: capsules pivoting at the shoulders */}
            <g className="rb-arm rb-arm-l">
              <rect x="14" y="216" width="86" height="30" rx="15" fill="url(#rb-limb)" />
              <circle cx="30" cy="231" r="21" fill="url(#rb-plush)" />
              <path d="M44 220 v22" className="rb-stitch" />
            </g>
            <g className="rb-arm rb-arm-r">
              <rect x="220" y="216" width="86" height="30" rx="15" fill="url(#rb-limb)" />
              <circle cx="290" cy="231" r="21" fill="url(#rb-plush)" />
              <path d="M276 220 v22" className="rb-stitch" />
            </g>

            {/* body */}
            <rect x="82" y="190" width="156" height="136" rx="64" fill="url(#rb-plush)" />
            <rect x="90" y="198" width="140" height="120" rx="58" className="rb-stitch" />
            <ellipse cx="124" cy="226" rx="30" ry="16" fill="url(#rb-shine)" />
            <rect x="118" y="236" width="84" height="56" rx="24" fill="url(#rb-belly)" />
            <rect x="124" y="242" width="72" height="44" rx="19" className="rb-stitch rb-stitch-light" />
            <g className="rb-chest">
              <circle cx="160" cy="264" r="14" fill="#3ee08a" filter="url(#rb-glow)" />
              <circle cx="160" cy="264" r="9" fill="#3ee08a" />
              <circle cx="157" cy="261" r="3" fill="#fff" opacity=".8" />
            </g>

            {/* head */}
            <g className="rb-head">
              <g className="rb-antenna">
                <path d="M160 62 V36" stroke="#9db4ae" strokeWidth="5" strokeLinecap="round" fill="none" />
                <circle cx="160" cy="28" r="14" fill="#3ee08a" filter="url(#rb-glow)" />
                <circle cx="160" cy="28" r="9" fill="#3ee08a" />
                <circle cx="157" cy="25" r="3" fill="#fff" opacity=".8" />
              </g>
              <circle cx="64" cy="134" r="19" fill="#3ee08a" />
              <circle cx="256" cy="134" r="19" fill="#3ee08a" />
              <circle cx="64" cy="134" r="10" fill="#1c5a45" />
              <circle cx="256" cy="134" r="10" fill="#1c5a45" />
              <rect x="68" y="58" width="184" height="152" rx="68" fill="url(#rb-plush)" />
              <rect x="76" y="66" width="168" height="136" rx="62" className="rb-stitch" />
              <ellipse cx="116" cy="84" rx="42" ry="17" fill="url(#rb-shine)" />
              <rect x="90" y="90" width="140" height="90" rx="40" fill="url(#rb-visor)" />
              <rect x="90" y="90" width="140" height="90" rx="40" fill="none" stroke="#3ee08a" strokeOpacity=".25" strokeWidth="2" />
              <g className="rb-eye rb-eye-l">
                <ellipse cx="134" cy="128" rx="14" ry="18" fill="#3ee08a" filter="url(#rb-glow)" />
                <ellipse cx="134" cy="128" rx="11" ry="15" fill="#6df5b0" />
                <circle cx="130" cy="122" r="4" fill="#fff" />
              </g>
              <g className="rb-eye rb-eye-r">
                <ellipse cx="186" cy="128" rx="14" ry="18" fill="#3ee08a" filter="url(#rb-glow)" />
                <ellipse cx="186" cy="128" rx="11" ry="15" fill="#6df5b0" />
                <circle cx="182" cy="122" r="4" fill="#fff" />
              </g>
              <path d="M144 158 Q160 172 176 158" stroke="#6df5b0" strokeWidth="5" strokeLinecap="round" fill="none" />
              <ellipse cx="108" cy="160" rx="10" ry="6" fill="#ff8fb1" opacity=".55" />
              <ellipse cx="212" cy="160" rx="10" ry="6" fill="#ff8fb1" opacity=".55" />
            </g>
          </g>
        </g>
      </svg>
    </div>
  );
}
