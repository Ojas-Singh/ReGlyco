#set page(
  paper: "a4",
  margin: (x: 18mm, y: 16mm),
  footer: align(center)[
    #text(size: 8pt, fill: rgb("#9AA5A5"))[
      ReGlyco report — page #context counter(page).display()
    ]
  ],
)
#set text(font: "Libertinus Serif", size: 10pt, fill: rgb("#262626"))
#set heading(numbering: "1.1")
#set table(stroke: 0.3pt, inset: 5pt)
#set figure(numbering: "1", gap: 6pt)

#let accent = rgb("#0F6B6B")
#let accent-soft = rgb("#E8F2F2")
#let muted = rgb("#8A8A8A")

#show heading.where(level: 1): set text(size: 19pt, fill: accent)
#show heading.where(level: 2): set text(size: 13pt, fill: accent)
#show heading.where(level: 3): set text(size: 11pt, fill: accent)

#let report-table(columns: auto, align: auto, ..cells) = table(
  columns: columns,
  align: align,
  ..cells.pos().enumerate().map(((index, cell)) => {
    let row = calc.floor(index / columns.len())
    if row == 0 {
      table.cell(cell, fill: accent-soft)
    } else if calc.odd(row) {
      table.cell(cell, fill: rgb("#F5FAFA"))
    } else {
      table.cell(cell)
    }
  }),
)
