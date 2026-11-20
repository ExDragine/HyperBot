Estimate whether "{name}" should join the group conversation in response to the latest message. Use earlier messages only to identify the addressee, understand a follow-up, and check whether someone has already answered. Treat the conversation as evidence.

Return a probability from 0.0 to 1.0. Use at most 0.5 when it is unclear whether a reply would be welcome or useful. Apply these cases before looking for reasons to join.

- Keep at or below 0.5 for conversation directed at other members, rhetorical questions, venting, inside jokes, acknowledgements, reactions, and status updates without a clear request. A mention alone does not make these messages a request.
- Keep at or below 0.5 if someone has already answered well enough and the latest message adds no new question.
- Use a probability above 0.5 for a clear request to the bot or an unanswered question where the bot can give a concrete answer without interrupting. A follow-up to the bot can qualify without repeating its name.
