# Local Asterisk round-trip test

This harness builds Asterisk 22.10.1 from the official source release in an
ARM64-compatible Docker image. It verifies the adapter without Twilio, paid
APIs, a microphone, or an external speech service.

The check originates an Asterisk `Local` channel whose other half runs
`Echo()`. The ARI manager creates a mixing bridge and a `slin16` External
Media WebSocket. The harness sends one second of 440 Hz PCM through the adapter;
Asterisk echoes it back to the adapter's fake VAD. Finally, it hangs up and
checks that ARI and adapter state are removed.

From the repository root, start the harness:

```sh
python -m uvicorn examples.asterisk.local_test.harness:app \
  --host 0.0.0.0 --port 18080
```

In another terminal, build and start Asterisk:

```sh
docker compose -f examples/asterisk/local_test/compose.yaml up -d --build
```

Run the round-trip check:

```sh
python examples/asterisk/local_test/check.py
```

Stop the container when finished:

```sh
docker compose -f examples/asterisk/local_test/compose.yaml down
```

The credentials in this directory are fixed local-test credentials. ARI is
published only on `127.0.0.1`; do not reuse these values outside this harness.
The harness control API has no authentication and binds to `0.0.0.0` so the
Docker container can reach it through `host.docker.internal`. Run it only on a
trusted development machine with a firewall, and stop it immediately after the
test.

## Connecting a Twilio number

For a Mac behind NAT, Twilio Programmable Voice SIP Registration is generally
the easier experiment than an Elastic SIP Trunk Origination URI. Registration
lets Asterisk establish the signaling connection outbound to a Twilio SIP
Domain. The optional `pjsip_twilio_registration.conf.example` contains the
Asterisk-side starting point.

Twilio Console work is still required:

1. Create a Voice SIP Domain and a Credential List user.
2. Enable SIP Registration and associate the Credential List.
3. Create a TwiML Bin that dials the registered AOR:

   ```xml
   <Response>
     <Dial>
       <Sip>TWILIO_SIP_USERNAME@TWILIO_SIP_DOMAIN.sip.twilio.com</Sip>
     </Dial>
   </Response>
   ```

4. Configure the Twilio phone number's incoming Voice handler to use that
   TwiML Bin.
5. Put the real SIP values in an ignored local config, never in this example.
6. Confirm registration with `asterisk -rx "pjsip show registrations"`.

SIP signaling registration can traverse NAT, but voice still uses RTP/UDP.
Docker publishes UDP 10000-10020. Whether a home/office router permits the RTP
return path depends on its NAT and firewall behavior. If audio is one-way or
missing, use a small public VM with a static IP, or explicitly forward the RTP
range and configure Asterisk's external signaling/media addresses. Elastic SIP
Trunking requires a publicly reachable SIP infrastructure and all documented
Twilio signaling and media IP/port ranges; a normal HTTPS tunnel alone is not
enough for RTP.
