# eecs589-patel-schauer

See our final report here: [Final Report](./files/eecs589_final_report.pdf).

## Abstract
Recent years have seen rapid development and adoption of encrypted DNS techniques, including DNS over TLS (DoT), DNS over HTTPS (DoH), and most recently, DNS over QUIC (DoQ). These techniques aim to protect user privacy by encrypting DNS queries and responses; however, previous studies have shown that traffic analysis attacks can still reasonably deduce the content for encrypted DoT and DoH traffic. In this study, we show that DoQ is also susceptible to traffic analysis attacks. We develop a machine learning classifier that infers DNS traffic based on patterns of packet size and timing. Our classifier achieves up to 98.4\% accuracy on our dataset of 200 websites. We then discuss different mitigation strategies, and show that while padding can defend against traffic analysis attacks to a degree, disguising the timing information is more helpful in defense. Our results demonstrate that traffic analysis attacks remain a concern with DoQ, and highlight the importance of further research on effective strategies to mitigate such attacks.
